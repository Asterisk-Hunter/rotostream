"""End-to-end HTTP flow: upload, extract, prompt, track, overlay, export, delete."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import registry
from app.models.base import Direction, FrameResult, PromptSet, TrackerInfo, VideoObjectTracker
from marks import requires_ffmpeg

PROMPT = {"frame_index": 0, "points": [{"x": 25, "y": 48, "positive": True}]}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def upload_ready(client: TestClient, video_path) -> str:
    with video_path.open("rb") as handle:
        response = client.post(
            "/api/videos", files={"file": ("sample.mp4", handle, "video/mp4")}
        )
    assert response.status_code == 201, response.text
    payload = response.json()
    job = wait_for_job(client, payload["job_id"])
    assert job["status"] == "succeeded", job
    return payload["video"]["id"]


# ------------------------------------------------------------------ catalog
def test_health_reports_capabilities(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["ffmpeg"] is True
    assert payload["torch"] is True


def test_models_endpoint_reports_the_memory_tracker_as_implemented(client):
    models = {model["name"]: model for model in client.get("/api/models").json()}
    assert models["naive"]["implemented"] is True
    assert models["naive"]["is_default"] is True
    assert models["sam2_memory"]["implemented"] is True
    assert models["sam2_memory"]["uses_memory"] is True
    assert models["sam2_memory"]["trainable"] is True
    # The studio warns about a background-only prompt frame before a run only when
    # the selected tracker cannot accept one.
    assert models["naive"]["accepts_background_only_prompts"] is False
    assert models["sam2_memory"]["accepts_background_only_prompts"] is True


def test_empty_video_list(client):
    assert client.get("/api/videos").json() == []


def test_unknown_video_is_404(client):
    assert client.get("/api/videos/nope").status_code == 404
    assert client.delete("/api/videos/nope").status_code == 404


# ------------------------------------------------------------------- upload
def test_upload_rejects_an_unsupported_extension(client):
    response = client.post(
        "/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415
    assert ".mp4" in response.json()["detail"]


@requires_ffmpeg
def test_upload_then_extract_produces_readable_frames(client, sample_video):
    with sample_video.open("rb") as handle:
        response = client.post(
            "/api/videos", files={"file": ("sample.mp4", handle, "video/mp4")}
        )
    assert response.status_code == 201, response.text
    video_id = response.json()["video"]["id"]
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["progress"] == 1.0

    video = client.get(f"/api/videos/{video_id}").json()
    assert video["status"] == "ready"
    assert video["n_frames"] == 12
    assert (video["frame_width"], video["frame_height"]) == (128, 96)
    assert video["size_bytes"] > 0

    frame = client.get(f"/api/videos/{video_id}/frames/0")
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.headers["Cache-Control"].startswith("private,")
    assert client.get(f"/api/videos/{video_id}/thumbnail").status_code == 200

    assert client.get(f"/api/videos/{video_id}/frames/999").status_code == 404
    assert client.get(f"/api/videos/{video_id}/frames/-1").status_code == 404


@requires_ffmpeg
def test_listed_videos_include_the_upload(client, sample_video):
    video_id = upload_ready(client, sample_video)
    listing = client.get("/api/videos").json()
    assert [item["id"] for item in listing] == [video_id]


class TodoTracker(VideoObjectTracker):
    """Registered but not written yet: the API must fail fast rather than crash."""

    @classmethod
    def info(cls) -> TrackerInfo:
        return TrackerInfo(
            name="todo_model", description="placeholder tracker", implemented=False
        )

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        raise NotImplementedError

    def set_video(self, frames) -> None:  # noqa: ANN001
        raise NotImplementedError

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        raise NotImplementedError

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        raise NotImplementedError


# ------------------------------------------------------------------ tracking
@requires_ffmpeg
def test_unimplemented_tracker_fails_fast_with_a_clear_409(client, monkeypatch, sample_video):
    # Every shipped tracker is implemented now, so the fail-fast path is covered by
    # registering a placeholder under an ephemeral key: a half-written plugin must
    # answer with 409 up front instead of dying on the worker thread.
    monkeypatch.setitem(registry._OVERRIDES, "todo_model", "test_api:TodoTracker")

    video_id = upload_ready(client, sample_video)
    response = client.post(
        f"/api/videos/{video_id}/track",
        json={"prompts": [PROMPT], "model": "todo_model"},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "todo_model" in detail and "available tracker" in detail


def test_unknown_tracker_is_404(client):
    response = client.post(
        "/api/videos/anything/track", json={"prompts": [PROMPT], "model": "gpt-9"}
    )
    assert response.status_code == 404


@requires_ffmpeg
def test_preview_returns_an_overlay_png_with_mask_stats(client, sample_video):
    video_id = upload_ready(client, sample_video)
    response = client.post(f"/api/videos/{video_id}/preview", json={"prompt": PROMPT})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert int(response.headers["X-Mask-Area"]) > 0
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


@requires_ffmpeg
def test_preview_rejects_an_out_of_range_frame(client, sample_video):
    video_id = upload_ready(client, sample_video)
    payload = {"prompt": {"frame_index": 500, "points": [{"x": 5, "y": 5}]}}
    assert client.post(f"/api/videos/{video_id}/preview", json=payload).status_code == 422


@requires_ffmpeg
def test_full_track_overlay_export_download_flow(client, sample_video):
    video_id = upload_ready(client, sample_video)

    started = client.post(f"/api/videos/{video_id}/track", json={"prompts": [PROMPT]})
    assert started.status_code == 202, started.text
    job = wait_for_job(client, started.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["n_tracked"] == 12
    assert job["result"]["n_frames"] == 12

    sessions = client.get(f"/api/videos/{video_id}/sessions").json()
    assert len(sessions) == 1
    latest = client.get(f"/api/videos/{video_id}/sessions/latest").json()
    assert latest["model"] == "naive"
    assert latest["n_tracked"] == 12
    assert len(latest["scores"]) == 12
    assert latest["prompt_frames"] == [0]
    assert latest["mean_score"] > 0

    # A mask exists for every frame, including frames after the prompt.
    for index in (0, 5, 11):
        mask = client.get(f"/api/videos/{video_id}/masks/{index}")
        assert mask.status_code == 200, mask.text
        assert mask.headers["content-type"] == "image/png"

    overlay = client.get(f"/api/videos/{video_id}/overlays/7")
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"
    assert int(overlay.headers["X-Mask-Area"]) > 0

    # Export, then download.
    export = client.post(f"/api/videos/{video_id}/exports", json={"kind": "mask_zip"})
    assert export.status_code == 202, export.text
    export_job = wait_for_job(client, export.json()["id"])
    assert export_job["status"] == "succeeded", export_job

    listing = client.get(f"/api/videos/{video_id}/exports").json()
    assert len(listing) == 1
    assert listing[0]["status"] == "succeeded"
    assert listing[0]["size_bytes"] > 0
    assert listing[0]["download_url"].endswith("/download")

    download = client.get(listing[0]["download_url"])
    assert download.status_code == 200
    assert len(download.content) == listing[0]["size_bytes"]
    assert download.content[:2] == b"PK", "mask_zip should be a real zip"

    # The artifact describes exactly what it contains, from the same facts the
    # renderer used, so the promise and the file cannot disagree.
    manifest = listing[0]["manifest"]
    assert manifest["kind"] == "mask_zip"
    assert manifest["n_frames"] == 12
    assert manifest["width"] > 0 and manifest["height"] > 0
    assert manifest["alpha"] is False
    assert any("no video and no audio" in note for note in manifest["notes"])
    assert any("frames have a mask" in note for note in manifest["notes"])


@requires_ffmpeg
def test_session_records_the_prompts_and_review_quality(client, sample_video):
    """A session must be able to explain how its masks were made and how good they are."""
    video_id = upload_ready(client, sample_video)
    boxed = {"frame_index": 3, "points": [], "box": {"x0": 10, "y0": 30, "x1": 60, "y1": 70}}
    started = client.post(
        f"/api/videos/{video_id}/track", json={"prompts": [PROMPT, boxed]}
    )
    assert started.status_code == 202, started.text
    assert wait_for_job(client, started.json()["id"])["status"] == "succeeded"

    session = client.get(f"/api/videos/{video_id}/sessions/latest").json()
    assert session["prompt_frames"] == [0, 3]
    assert [prompt["frame_index"] for prompt in session["prompts"]] == [0, 3]
    assert session["prompts"][0]["points"][0]["positive"] is True
    assert session["prompts"][1]["box"]["x1"] == 60

    quality = session["quality"]
    assert quality["n_frames"] == 12
    assert 0.0 <= quality["coverage"] <= 1.0
    assert quality["absent_frames"] == session["absent_frames"]
    assert quality["problem_frames"] == sorted(
        set(quality["absent_frames"]) | set(quality["low_confidence_frames"])
    )
    assert isinstance(quality["sound"], bool)
    assert session["sound"] == quality["sound"]
    # The flat fields are the same numbers, not the schema's hopeful defaults.
    assert session["coverage"] == quality["coverage"]
    assert session["low_confidence_frames"] == quality["low_confidence_frames"]
    assert session["background_only_frames"] == quality["background_only_frames"]
    # Prompted frames are the user's own instruction, so they are never reported as
    # low confidence.
    assert not set(quality["low_confidence_frames"]) & set(session["prompt_frames"])


@requires_ffmpeg
def test_source_audio_survives_a_video_export(client, sample_video, tmp_path):
    """An "edited clip" of a clip with sound must not come back silent."""
    import subprocess

    with_audio = tmp_path / "with-audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(sample_video), "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(with_audio),
        ],
        check=True,
        capture_output=True,
    )
    video_id = upload_ready(client, with_audio)
    assert client.get(f"/api/videos/{video_id}").json()["has_audio"] is True

    started = client.post(f"/api/videos/{video_id}/track", json={"prompts": [PROMPT]})
    assert wait_for_job(client, started.json()["id"])["status"] == "succeeded"
    export = client.post(
        f"/api/videos/{video_id}/exports",
        json={"kind": "replace_bg", "background": "black"},
    )
    assert export.status_code == 202, export.text
    assert wait_for_job(client, export.json()["id"])["status"] == "succeeded"

    listing = client.get(f"/api/videos/{video_id}/exports").json()
    manifest = listing[0]["manifest"]
    assert manifest["audio_preserved"] is True
    assert any("copied from the source" in note for note in manifest["notes"])
    assert any("Background: solid black" in note for note in manifest["notes"])

    download = client.get(listing[0]["download_url"])
    assert download.status_code == 200
    artifact = tmp_path / "edited.mp4"
    artifact.write_bytes(download.content)
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(artifact)],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "audio" in streams, f"export dropped the source audio: {streams!r}"


@requires_ffmpeg
def test_export_before_tracking_is_a_clear_409(client, sample_video):
    video_id = upload_ready(client, sample_video)
    response = client.post(f"/api/videos/{video_id}/exports", json={"kind": "mask_zip"})
    assert response.status_code == 409
    assert "track" in response.json()["detail"].lower()


@requires_ffmpeg
def test_re_prompting_creates_a_second_session(client, sample_video):
    video_id = upload_ready(client, sample_video)
    for _ in range(2):
        response = client.post(f"/api/videos/{video_id}/track", json={"prompts": [PROMPT]})
        assert wait_for_job(client, response.json()["id"])["status"] == "succeeded"
    assert len(client.get(f"/api/videos/{video_id}/sessions").json()) == 2


@requires_ffmpeg
def test_an_old_session_still_reports_what_is_missing(client, sample_video):
    """A run written before the review summary existed must not look complete.

    These sessions are the ones already sitting in the workspace, and the studio's
    whole promise is that it never claims a mask when frames have none.
    """
    import numpy as np

    from app import masks as mask_utils
    from app.storage import get_workspace

    video_id = upload_ready(client, sample_video)
    workspace = get_workspace()
    session_id = workspace.create_session(video_id)
    missing = [4, 5, 6]
    scores = [
        {
            "frame_index": index,
            "score": 0.9 if index not in missing else 0.0,
            "object_present": index not in missing,
            "prompted": index == 0,
        }
        for index in range(12)
    ]
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    for index in range(12):
        mask_utils.save_mask(workspace.mask_path(video_id, session_id, index), mask)
    workspace.write_session_meta(
        video_id,
        session_id,
        {
            "id": session_id, "video_id": video_id, "model": "naive",
            "created_at": "2026-01-01T00:00:00+00:00", "status": "succeeded",
            "n_tracked": 12, "n_frames": 12, "prompt_frames": [0],
            "mean_score": 0.9, "absent_frames": missing, "scores": scores,
        },
    )

    session = client.get(f"/api/videos/{video_id}/sessions/{session_id}").json()
    assert session["quality"]["n_masked"] == 9
    assert session["coverage"] == 0.75
    assert session["sound"] is False
    assert session["absent_frames"] == missing
    assert session["quality"]["problem_frames"] == missing
    listed = client.get(f"/api/videos/{video_id}/sessions").json()
    assert [(item["coverage"], item["sound"]) for item in listed] == [(0.75, False)]

    # ... and an export of that session says so, rather than shipping the file quietly.
    export = client.post(
        f"/api/videos/{video_id}/exports",
        json={"kind": "mask_zip", "session_id": session_id},
    )
    assert export.status_code == 202, export.text
    assert wait_for_job(client, export.json()["id"])["status"] == "succeeded"
    manifest = client.get(f"/api/videos/{video_id}/exports").json()[0]["manifest"]
    assert any("9 of 12 frames have a mask" in note for note in manifest["notes"])
    assert any("review them in the timeline" in note for note in manifest["notes"])


# --------------------------------------------------------------------- jobs
def test_unknown_job_endpoints_are_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404


@requires_ffmpeg
def test_jobs_can_be_filtered_by_video(client, sample_video):
    video_id = upload_ready(client, sample_video)
    client.post(f"/api/videos/{video_id}/track", json={"prompts": [PROMPT]})
    assert client.get(f"/api/jobs?video_id={video_id}").json()
    assert client.get("/api/jobs?video_id=other").json() == []


# ------------------------------------------------------------------ deletion
@requires_ffmpeg
def test_deleting_a_video_removes_its_data(client, sample_video):
    video_id = upload_ready(client, sample_video)
    assert client.delete(f"/api/videos/{video_id}").status_code == 204
    assert client.get(f"/api/videos/{video_id}").status_code == 404
    assert client.get(f"/api/videos/{video_id}/frames/0").status_code == 404
