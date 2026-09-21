"""Videos: upload, probe/extract, serve frames, masks, overlays and sessions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .. import masks as mask_utils
from ..jobs import get_job_manager
from ..schemas import SessionOut, UploadOut, VideoOut, VideoStatus
from ..settings import Settings, get_settings
from ..storage import Workspace, get_workspace
from ..video import FFmpegError, extract_frames, probe, read_frame

router = APIRouter(prefix="/api/videos", tags=["videos"])

CHUNK_BYTES = 1024 * 1024
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


# --------------------------------------------------------------------- helpers
def require_video(video_id: str) -> dict:
    try:
        return get_workspace().read_meta(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown video {video_id}") from exc


def require_ready_video(video_id: str) -> dict:
    meta = require_video(video_id)
    if meta.get("status") != VideoStatus.READY.value:
        raise HTTPException(
            status_code=409,
            detail=f"video is {meta.get('status')!r}, not ready (error: {meta.get('error')})",
        )
    return meta


def to_video_out(meta: dict) -> VideoOut:
    """Meta records accumulate keys over time; keep only the ones the schema knows."""
    payload = {key: meta[key] for key in VideoOut.model_fields if key in meta}
    return VideoOut(**payload)


def _resolve_session(workspace: Workspace, video_id: str, session_id: str | None) -> str:
    resolved = session_id or workspace.latest_session_id(video_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="no tracking session exists for this video")
    return resolved


def _read_frame(workspace: Workspace, video_id: str, index: int) -> np.ndarray:
    try:
        return read_frame(workspace.frames_dir(video_id), index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _extract_body(ctx, video_id: str, source: Path, settings: Settings) -> dict:
    """Decode to JPEG frames and record geometry. Runs on the job thread."""
    workspace = get_workspace()
    try:
        ctx.progress(0.05, "probing")
        info = probe(source)
        workspace.update_meta(
            video_id,
            width=info.width,
            height=info.height,
            fps=round(info.fps, 6),
            duration_s=round(info.duration_s, 3),
            n_frames=info.n_frames,
            has_audio=info.has_audio,
        )

        ctx.progress(0.2, "extracting frames")
        n_frames, width, height = extract_frames(
            source,
            workspace.frames_dir(video_id),
            long_side=settings.long_side,
            quality=settings.jpeg_quality,
            max_frames=settings.max_frames,
        )
        workspace.update_meta(
            video_id,
            status=VideoStatus.READY.value,
            n_frames=n_frames,
            frame_width=width,
            frame_height=height,
            error=None,
        )
        ctx.progress(1.0, f"{n_frames} frames at {width}x{height}")
        return {"video_id": video_id, "n_frames": n_frames, "frame_width": width, "frame_height": height}
    except FFmpegError as exc:
        workspace.update_meta(video_id, status=VideoStatus.FAILED.value, error=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - always record why it failed
        workspace.update_meta(video_id, status=VideoStatus.FAILED.value, error=str(exc))
        raise


# ---------------------------------------------------------------------- upload
@router.post("", response_model=UploadOut, status_code=201, summary="Upload a video")
async def upload_video(file: UploadFile = File(...)) -> UploadOut:
    settings = get_settings()
    workspace = get_workspace()

    filename = file.filename or "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extension_set:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported extension {suffix!r}; allowed: {sorted(settings.allowed_extension_set)}",
        )

    video_id = workspace.create_video(filename)
    destination = workspace.video_dir(video_id) / f"source{suffix}"

    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(CHUNK_BYTES):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds ROTOSTREAM_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)",
                    )
                handle.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="uploaded file is empty")
    except HTTPException:
        workspace.delete_video(video_id)
        raise
    finally:
        await file.close()

    workspace.update_meta(
        video_id,
        size_bytes=written,
        status=VideoStatus.EXTRACTING.value,
        source=f"source{suffix}",
    )

    manager = get_job_manager()
    job = manager.create("extract", video_id=video_id)
    manager.submit(job, lambda ctx: _extract_body(ctx, video_id, destination, settings))

    return UploadOut(video=to_video_out(workspace.read_meta(video_id)), job_id=job.id)


# ----------------------------------------------------------------------- reads
@router.get("", response_model=list[VideoOut], summary="List videos")
def list_videos() -> list[VideoOut]:
    return [to_video_out(meta) for meta in get_workspace().list_videos()]


@router.get("/{video_id}", response_model=VideoOut, summary="Video details")
def get_video(video_id: str) -> VideoOut:
    return to_video_out(require_video(video_id))


@router.delete("/{video_id}", status_code=204, summary="Delete a video and all its data")
def delete_video(video_id: str) -> Response:
    if not get_workspace().delete_video(video_id):
        raise HTTPException(status_code=404, detail=f"unknown video {video_id}")
    return Response(status_code=204)


# ---------------------------------------------------------------------- frames
@router.get("/{video_id}/frames/{index}", summary="Working-resolution frame JPEG")
def get_frame(video_id: str, index: int) -> FileResponse:
    meta = require_video(video_id)
    workspace = get_workspace()
    if not 0 <= index < int(meta.get("n_frames") or 0):
        raise HTTPException(status_code=404, detail=f"frame {index} out of range")
    path = workspace.frames_dir(video_id) / f"{index:06d}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"frame {index} not extracted")
    return FileResponse(path, media_type="image/jpeg", headers=IMMUTABLE)


@router.get("/{video_id}/thumbnail", summary="First frame, for the project list")
def get_thumbnail(video_id: str) -> FileResponse:
    return get_frame(video_id, 0)


# ------------------------------------------------------------------ masks/overlay
@router.get("/{video_id}/masks/{index}", summary="Binary mask PNG")
def get_mask(video_id: str, index: int, session_id: str | None = Query(default=None)) -> FileResponse:
    require_video(video_id)
    workspace = get_workspace()
    resolved = _resolve_session(workspace, video_id, session_id)
    path = workspace.mask_path(video_id, resolved, index)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"frame {index} has no mask in session {resolved}",
        )
    headers = dict(IMMUTABLE) if session_id else {"Cache-Control": "no-cache"}
    return FileResponse(path, media_type="image/png", headers=headers)


@router.get("/{video_id}/overlays/{index}", summary="Frame with the mask tinted on top")
def get_overlay(
    video_id: str,
    index: int,
    session_id: str | None = Query(default=None),
    color: str = Query(default="56,189,248", description="Overlay colour as r,g,b"),
    alpha: float = Query(default=0.38, ge=0.0, le=1.0),
) -> Response:
    """Rendered server-side so the browser just stacks an <img> - no canvas work."""
    meta = require_video(video_id)
    workspace = get_workspace()
    if not 0 <= index < int(meta.get("n_frames") or 0):
        raise HTTPException(status_code=404, detail=f"frame {index} out of range")
    resolved = _resolve_session(workspace, video_id, session_id)

    frame = _read_frame(workspace, video_id, index)
    height, width = frame.shape[:2]
    mask = mask_utils.load_mask_or_empty(
        workspace.mask_path(video_id, resolved, index), (height, width)
    )
    try:
        rgb = tuple(int(part) for part in color.split(","))
        if len(rgb) != 3 or any(not 0 <= part <= 255 for part in rgb):
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="color must be 'r,g,b' with 0-255") from exc

    png = mask_utils.overlay_png_bytes(frame, mask, color=rgb, alpha=alpha)
    headers = dict(IMMUTABLE) if session_id else {"Cache-Control": "no-cache"}
    headers["X-Mask-Area"] = str(int(mask.sum()))
    return Response(content=png, media_type="image/png", headers=headers)


# --------------------------------------------------------------------- sessions
@router.get("/{video_id}/sessions", response_model=list[SessionOut], summary="List sessions")
def list_sessions(video_id: str) -> list[SessionOut]:
    require_video(video_id)
    return [SessionOut(**meta) for meta in get_workspace().list_sessions(video_id)]


@router.get("/{video_id}/sessions/latest", response_model=SessionOut, summary="Most recent session")
def latest_session(video_id: str) -> SessionOut:
    require_video(video_id)
    workspace = get_workspace()
    session_id = workspace.latest_session_id(video_id)
    if session_id is None:
        raise HTTPException(status_code=404, detail="no tracking session exists for this video")
    return SessionOut(**workspace.read_session_meta(video_id, session_id))


@router.get("/{video_id}/sessions/{session_id}", response_model=SessionOut, summary="Session details")
def get_session(video_id: str, session_id: str) -> SessionOut:
    require_video(video_id)
    try:
        meta = get_workspace().read_session_meta(video_id, session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionOut(**meta)
