"""Storage and HTTP failure boundaries without downloading model weights."""
from __future__ import annotations

import subprocess
import pytest
from fastapi.testclient import TestClient

from app.jobs import JobCancelled, get_job_manager
from app.main import app
from app.schemas import BoxIn, PointIn
from app.storage import get_workspace, utcnow
from app.video import ExportInputs, FFmpegError, _run, run_export


def ready_video():
    workspace = get_workspace()
    video_id = workspace.create_video("test.mp4")
    workspace.update_meta(video_id, status="ready", n_frames=2,
                          frame_width=8, frame_height=8, fps=10)
    return workspace, video_id


def session(workspace, video_id, status):
    session_id = workspace.create_session(video_id)
    workspace.write_session_meta(video_id, session_id, {
        "id": session_id, "video_id": video_id, "model": "naive",
        "created_at": utcnow(), "status": status,
    })
    return session_id


@pytest.mark.parametrize("identifier", ["..", "../secret", "a/b", "a\\b", "C:\\secret", ""])
def test_workspace_rejects_traversal_identifiers(identifier):
    workspace = get_workspace()
    with pytest.raises(FileNotFoundError):
        workspace.video_dir(identifier)
    with pytest.raises(FileNotFoundError):
        workspace.session_dir("video", identifier)
    with pytest.raises(FileNotFoundError):
        workspace.export_meta_path("video", identifier)


def test_latest_session_ignores_failed_and_pending_sessions():
    workspace, video_id = ready_video()
    complete = session(workspace, video_id, "succeeded")
    session(workspace, video_id, "failed")
    session(workspace, video_id, "queued")
    assert workspace.latest_session_id(video_id) == complete


def test_restart_marks_interrupted_records_failed():
    workspace = get_workspace()
    video_id = workspace.create_video("test.mp4")
    pending = session(workspace, video_id, "running")
    workspace.write_export_meta(video_id, "export", {"id": "export", "status": "queued"})
    workspace.recover_interrupted()
    assert workspace.read_meta(video_id)["status"] == "failed"
    assert workspace.read_session_meta(video_id, pending)["status"] == "failed"
    assert workspace.read_export_meta(video_id, "export")["status"] == "failed"


def test_explicit_unknown_or_incomplete_session_cannot_be_exported():
    with TestClient(app) as client:
        workspace, video_id = ready_video()
        pending = session(workspace, video_id, "running")
        path = f"/api/videos/{video_id}/exports"
        assert client.post(path, json={"kind": "mask_zip", "session_id": "unknown"}).status_code == 404
        assert client.post(path, json={"kind": "mask_zip", "session_id": pending}).status_code == 409
        assert client.post(path, json={"kind": "mask_zip", "session_id": "../../outside"}).status_code == 422
        assert client.get(f"/api/videos/{video_id}/masks/0?session_id=../../outside").status_code == 404


def test_active_video_deletion_is_rejected():
    with TestClient(app) as client:
        workspace, video_id = ready_video()
        manager = get_job_manager()
        job = manager.create("track", video_id=video_id)
        assert client.delete(f"/api/videos/{video_id}").status_code == 409
        manager.cancel(job.id)
        assert client.delete(f"/api/videos/{video_id}").status_code == 204


def test_session_ids_cannot_be_reused_to_mutate_cached_masks():
    with TestClient(app) as client:
        workspace, video_id = ready_video()
        complete = session(workspace, video_id, "succeeded")
        response = client.post(f"/api/videos/{video_id}/track", json={
            "session_id": complete, "prompts": [{"frame_index": 0, "points": [{"x": 1, "y": 1}]}],
        })
        assert response.status_code == 409
        assert workspace.read_session_meta(video_id, complete)["status"] == "succeeded"


def test_queue_full_returns_retryable_503_without_creating_records(monkeypatch):
    monkeypatch.setenv("ROTOSTREAM_MAX_PENDING_JOBS", "1")
    with TestClient(app) as client:
        workspace, video_id = ready_video()
        get_job_manager().create("reserved")
        response = client.post(f"/api/videos/{video_id}/track", json={
            "prompts": [{"frame_index": 0, "points": [{"x": 1, "y": 1}]}],
        })
        assert response.status_code == 503 and response.headers["Retry-After"] == "5"
        assert workspace.list_sessions(video_id) == []


def test_readiness_reports_dependency_failure_without_loading_weights(monkeypatch):
    from app.routers import health
    monkeypatch.setattr(health, "ffmpeg_available", lambda: False)
    with TestClient(app) as client:
        response = client.get("/api/ready")
        assert response.status_code == 503
        assert response.json()["checks"] == {"ffmpeg": False, "storage": True, "tracker": True}


def test_failed_export_removes_partial_artifact(tmp_path, monkeypatch):
    from app import video
    out = tmp_path / "artifact.zip"
    def fail(inp, progress):
        inp.out_path.write_bytes(b"partial")
        raise JobCancelled()
    monkeypatch.setitem(video.EXPORTERS, "mask_zip", fail)
    inputs = ExportInputs(tmp_path, tmp_path, 2, 10, 8, 8, out)
    with pytest.raises(JobCancelled):
        run_export("mask_zip", inputs, lambda *args: None)
    assert not out.exists()


def test_media_subprocess_has_a_finite_deadline(monkeypatch):
    def timeout(cmd, **kwargs):
        assert kwargs["timeout"] > 0
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(FFmpegError, match="time limit"):
        _run(["ffmpeg"])


def test_prompt_coordinates_reject_nonfinite_values_and_flat_boxes():
    with pytest.raises(ValueError):
        PointIn(x=float("nan"), y=1)
    with pytest.raises(ValueError):
        BoxIn(x0=0, x1=4, y0=2, y1=2)


def test_decode_rejects_oversized_source_before_starting_ffmpeg(tmp_path, monkeypatch):
    from app import video
    monkeypatch.setattr(video, "probe", lambda source: video.VideoProbe(16384, 16384, 30, 1, 30, False))
    calls = []
    monkeypatch.setattr(video, "_run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(FFmpegError, match="MAX_SOURCE_PIXELS"):
        video.extract_frames(tmp_path / "source.mp4", tmp_path / "frames")
    assert not calls


def test_failed_upload_cleans_record_and_releases_admission(monkeypatch):
    from pathlib import Path
    original = Path.open
    def fail_source(path, *args, **kwargs):
        if path.name == "source.mp4":
            raise OSError("fixture disk full")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_source)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/videos", files={"file": ("sample.mp4", b"video", "video/mp4")})
        assert response.status_code == 500
        assert get_workspace().list_videos() == []
        assert all(job.done for job in get_job_manager().list())


def test_queued_export_cancellation_updates_durable_record():
    with TestClient(app) as client:
        workspace, video_id = ready_video()
        complete = session(workspace, video_id, "succeeded")
        manager = get_job_manager()
        with manager.compute_slot():
            response = client.post(f"/api/videos/{video_id}/exports", json={
                "kind": "mask_zip", "session_id": complete,
            })
            assert response.status_code == 202
            job_id = response.json()["id"]
            assert client.delete(f"/api/videos/{video_id}/exports/{workspace.list_exports(video_id)[0]['id']}").status_code == 409
            assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
            record = workspace.list_exports(video_id)[0]
            assert record["status"] == "cancelled"
            assert workspace.export_artifact(video_id, record["id"]) is None
