"""End-to-end HTTP flow: upload, extract, prompt, track, overlay, export, delete."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
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


def test_models_endpoint_marks_the_stub_as_unimplemented(client):
    models = {model["name"]: model for model in client.get("/api/models").json()}
    assert models["naive"]["implemented"] is True
    assert models["naive"]["is_default"] is True
    assert models["sam2_memory"]["implemented"] is False
    assert models["sam2_memory"]["uses_memory"] is True
    assert models["sam2_memory"]["trainable"] is True


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
    assert client.get(f"/api/videos/{video_id}/thumbnail").status_code == 200

    assert client.get(f"/api/videos/{video_id}/frames/999").status_code == 404
    assert client.get(f"/api/videos/{video_id}/frames/-1").status_code == 404


@requires_ffmpeg
def test_listed_videos_include_the_upload(client, sample_video):
    video_id = upload_ready(client, sample_video)
    listing = client.get("/api/videos").json()
    assert [item["id"] for item in listing] == [video_id]


# ------------------------------------------------------------------ tracking
@requires_ffmpeg
def test_stub_tracker_gives_a_clear_409(client, sample_video):
    video_id = upload_ready(client, sample_video)
    response = client.post(
        f"/api/videos/{video_id}/track",
        json={"prompts": [PROMPT], "model": "sam2_memory"},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "sam2_memory" in detail and "sam2_memory.py" in detail


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
