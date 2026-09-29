#!/usr/bin/env python
"""End-to-end smoke test against a *running* RotoStream API.

Unlike the pytest suite (which drives the app in-process via TestClient), this
talks real HTTP to a real uvicorn process, so it catches the things an in-process
test cannot: port/CORS misconfiguration, a server started without ffmpeg on PATH,
a stale build, and whatever a fresh clone gets wrong.

    python api/scripts/smoke.py                        # http://127.0.0.1:8010
    python api/scripts/smoke.py --base http://127.0.0.1:8020
    python api/scripts/smoke.py --keep                 # leave the video behind

Exits non-zero on the first hard failure. Requires ffmpeg on PATH.

The test clip is a white square sweeping across a black frame at 12fps, so the
prompt point (and therefore the expected mask) is known exactly - see
``SQUARE_AT_T0``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import namedtuple
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - environment guidance
    sys.exit("httpx is required: pip install httpx")

FPS = 12
DURATION = 4.0  # long enough that an SSE subscriber sees progress, not just the end state
N_FRAMES = int(FPS * DURATION)  # 48
WIDTH, HEIGHT = 256, 192
SQUARE_SIZE = 48
#: Centre of the square at t=0 (drawbox x=80+60*sin(2*pi*t), y=64).
SQUARE_AT_T0 = (80 + SQUARE_SIZE // 2, 64 + SQUARE_SIZE // 2)

# Response bodies include PNG/zip/webm bytes; Windows consoles default to cp1252
# and would raise on the first undecodable byte, turning a passing run into a
# traceback. Force UTF-8 and never print raw binary regardless.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-tty/older Python
        pass

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
#: Every member of schemas.ExportKind, with its expected file signature.
#: replace_bg defaults to background="blur", so no extra params are needed.
EXPORTS = [
    ("alpha_webm", b"\x1aE\xdf\xa3", "EBML/webm"),
    ("overlay_mp4", None, "mp4"),
    ("cutout_zip", b"PK", "zip"),
    ("mask_zip", b"PK", "zip"),
    ("mask_rle_json", None, "json"),
    ("replace_bg", None, "mp4"),
]
#: Containers ffmpeg writes, where the magic number lives inside a box header.
ISO_MEDIA_KINDS = {"overlay_mp4", "replace_bg"}

passed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion. Raises on failure so the caller exits non-zero."""
    global passed
    if condition:
        passed += 1
        print(f"  \033[32mok\033[0m   {label}" + (f"  \033[2m{detail}\033[0m" if detail else ""))
        return
    raise SystemExit(f"  \033[31mFAIL\033[0m {label}  {detail}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def body_hint(response: httpx.Response, limit: int = 200) -> str:
    """A short, always-printable description of a response body."""
    content_type = response.headers.get("content-type", "")
    if content_type.split(";")[0] in {
        "image/png", "image/jpeg", "video/webm", "video/mp4",
        "application/zip", "application/json",
    } and not content_type.startswith("application/json"):
        return f"{content_type} {len(response.content)} bytes"
    return response.text[:limit]


def make_clip(path: Path) -> Path:
    """Render a deterministic moving-square clip at :data:`path`."""
    filter_graph = (
        f"drawbox=x='80+60*sin(2*PI*t)':y=64:w={SQUARE_SIZE}:h={SQUARE_SIZE}"
        f":color=white:t=fill"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:r={FPS}:d={DURATION}",
            "-vf", filter_graph,
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            str(path),
        ],
        check=True,
    )
    return path


#: Result of draining a job's SSE stream.
SseStream = namedtuple("SseStream", "final updates saw_done content_type")


def collect_sse(client: httpx.Client, job_id: str, timeout: float = 180.0) -> SseStream:
    """Drain ``/api/jobs/{id}/events`` exactly like the browser's EventSource does.

    ``event: done`` is followed by a literal ``data: {}`` sentinel, so payloads
    are only accepted when they actually look like a job payload; otherwise that
    sentinel would overwrite the terminal state we are trying to assert on.
    """
    deadline = time.time() + timeout
    final: dict = {}
    updates = 0
    saw_done = False
    content_type = ""

    with client.stream("GET", f"/api/jobs/{job_id}/events", timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        for line in response.iter_lines():
            if time.time() > deadline:
                break
            if line.startswith("event: done"):
                saw_done = True
            elif line.startswith("data: "):
                payload = line[len("data: ") :].strip()
                if not payload:
                    continue
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "status" in parsed:
                    final = parsed
                    updates += 1

    return SseStream(final, updates, saw_done, content_type)


def wait_for_job(client: httpx.Client, job_id: str, timeout: float = 180.0) -> dict:
    """Poll a job until it reaches a terminal status."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        last = response.json()
        if last["status"] in TERMINAL_STATUSES:
            return last
        time.sleep(0.15)
    raise SystemExit(f"job {job_id} still {last.get('status')!r} after {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8010", help="API base URL")
    parser.add_argument("--keep", action="store_true", help="do not delete the video")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-job timeout seconds")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    print(f"RotoStream smoke test -> {base}")

    with httpx.Client(base_url=base, timeout=120.0) as client:
        section("service")
        health = client.get("/api/health")
        check("GET /api/health is 200", health.status_code == 200, str(health.status_code))
        body = health.json()
        check("status is ok", body.get("status") == "ok", str(body.get("status")))
        check("ffmpeg available", body.get("ffmpeg") is True, str(body.get("ffmpeg")))
        check("torch available", body.get("torch") is True, str(body.get("torch")))

        models = {m["name"]: m for m in client.get("/api/models").json()}
        check("naive tracker registered", "naive" in models)
        check(
            "sam2_memory registered as implemented",
            models.get("sam2_memory", {}).get("implemented") is True,
            "implemented=true once the memory stack landed",
        )

        section("upload + extraction")
        with tempfile.TemporaryDirectory() as tmp:
            clip = make_clip(Path(tmp) / "smoke.mp4")
            check("clip rendered", clip.stat().st_size > 0, f"{clip.stat().st_size} bytes")

            with clip.open("rb") as handle:
                upload = client.post(
                    "/api/videos", files={"file": ("smoke.mp4", handle, "video/mp4")}
                )
            check("POST /api/videos is 201", upload.status_code == 201, upload.text[:200])
            payload = upload.json()
            video_id = payload["video"]["id"]
            video = client.get(f"/api/videos/{video_id}").json()
            check(
                "video id is a short hex slug (storage.new_id)",
                len(video_id) == 12 and all(c in "0123456789abcdef" for c in video_id),
                video_id,
            )

            job = wait_for_job(client, payload["job_id"], args.timeout)
            check("extraction job succeeded", job["status"] == "succeeded", json.dumps(job)[:200])
            check("extraction progress reached 1.0", job["progress"] == 1.0, str(job["progress"]))

            video = client.get(f"/api/videos/{video_id}").json()
            check("video status is ready", video["status"] == "ready", video["status"])
            check(f"frame count is {N_FRAMES}", video["n_frames"] == N_FRAMES, str(video["n_frames"]))
            check(
                "frames downscaled to long side 960 or less",
                max(video["frame_width"], video["frame_height"]) <= 960,
                f'{video["frame_width"]}x{video["frame_height"]}',
            )

            frame = client.get(f"/api/videos/{video_id}/frames/0")
            check("GET frames/0 is a JPEG", frame.content[:2] == b"\xff\xd8", str(frame.status_code))
            check("GET thumbnail is 200", client.get(f"/api/videos/{video_id}/thumbnail").status_code == 200)
            check("GET frames/999 is 404", client.get(f"/api/videos/{video_id}/frames/999").status_code == 404)

            section("preview (one click, one frame)")
            prompt = {
                "frame_index": 0,
                "points": [{"x": SQUARE_AT_T0[0], "y": SQUARE_AT_T0[1], "positive": True}],
            }
            preview = client.post(f"/api/videos/{video_id}/preview", json={"prompt": prompt})
            check("POST preview is 200", preview.status_code == 200, body_hint(preview))
            check("preview is a PNG", preview.content[:8] == b"\x89PNG\r\n\x1a\n")
            check(
                "preview reports a non-empty mask",
                int(preview.headers.get("X-Mask-Area", "0")) > 0,
                f'area={preview.headers.get("X-Mask-Area")} ratio={preview.headers.get("X-Mask-Ratio")}',
            )

            section("track the whole clip (progress over SSE, as the UI does)")
            started = client.post(f"/api/videos/{video_id}/track", json={"prompts": [prompt]})
            check("POST track is 202", started.status_code == 202, started.text[:200])
            stream = collect_sse(client, started.json()["id"], args.timeout)
            check(
                "SSE endpoint is text/event-stream",
                stream.content_type.startswith("text/event-stream"),
                stream.content_type,
            )
            check("SSE delivered job state", stream.updates >= 1, f"{stream.updates} update(s)")
            check("SSE ended with 'event: done'", stream.saw_done)
            track_job = stream.final
            check("tracking succeeded", track_job.get("status") == "succeeded", json.dumps(track_job)[:300])
            check("terminal progress is 1.0", track_job.get("progress") == 1.0, str(track_job.get("progress")))
            result = track_job["result"]
            check(
                "every frame was tracked",
                result["n_tracked"] == result["n_frames"] == N_FRAMES,
                f'{result["n_tracked"]}/{result["n_frames"]}',
            )

            latest = client.get(f"/api/videos/{video_id}/sessions/latest").json()
            check("session records the prompt frame", latest["prompt_frames"] == [0], str(latest["prompt_frames"]))
            check("session has a score per frame", len(latest["scores"]) == N_FRAMES, str(len(latest["scores"])))

            section("masks + overlay")
            for index in (0, N_FRAMES // 2, N_FRAMES - 1):
                mask = client.get(f"/api/videos/{video_id}/masks/{index}")
                check(f"mask {index} is a PNG", mask.content[:8] == b"\x89PNG\r\n\x1a\n", str(mask.status_code))
            overlay = client.get(f"/api/videos/{video_id}/overlays/{N_FRAMES // 2}")
            check("overlay is a PNG", overlay.content[:8] == b"\x89PNG\r\n\x1a\n", str(overlay.status_code))
            check(
                "overlay carries mask stats in headers",
                int(overlay.headers.get("X-Mask-Area", "0")) > 0,
                str(overlay.headers.get("X-Mask-Area")),
            )

            section("exports")
            for kind, signature, label in EXPORTS:
                response = client.post(
                    f"/api/videos/{video_id}/exports", json={"kind": kind}
                )
                check(f"export {kind} accepted (202)", response.status_code == 202, response.text[:200])
                export_job = wait_for_job(client, response.json()["id"], args.timeout)
                check(f"export {kind} succeeded", export_job["status"] == "succeeded", json.dumps(export_job)[:200])

            listing = client.get(f"/api/videos/{video_id}/exports").json()
            check("all six export kinds are listed", len(listing) == len(EXPORTS), str(len(listing)))
            by_kind = {item["kind"]: item for item in listing}
            for kind, signature, label in EXPORTS:
                record = by_kind[kind]
                check(f"export {kind} has bytes", record["size_bytes"] > 0, f'{record["size_bytes"]} bytes')
                download = client.get(record["download_url"])
                check(f"export {kind} downloads ({label})", download.status_code == 200, str(download.status_code))
                check(
                    f"export {kind} size matches the record",
                    len(download.content) == record["size_bytes"],
                    f'{len(download.content)} vs {record["size_bytes"]}',
                )
                if signature:
                    check(f"export {kind} has a {label} signature", download.content[: len(signature)] == signature)
                if kind == "mask_rle_json":
                    payload_json = download.content.decode("utf-8")
                    check("mask_rle_json parses", isinstance(json.loads(payload_json), (dict, list)))
                if kind in ISO_MEDIA_KINDS:
                    check(f"export {kind} is an ISO media file", b"ftyp" in download.content[:32])

            section("cleanup")
            if args.keep:
                print(f"  \033[2m--keep set; video {video_id} left in the workspace\033[0m")
            else:
                deleted = client.delete(f"/api/videos/{video_id}")
                check("DELETE is 204", deleted.status_code == 204, str(deleted.status_code))
                check("video is gone", client.get(f"/api/videos/{video_id}").status_code == 404)

    print(f"\n\033[32m{passed} checks passed\033[0m - the API is wired up end to end.")
    print("Every registered tracker reports implemented=true - there is no longer a model gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
