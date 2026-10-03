"""Render deliverables from a completed tracking session."""
from __future__ import annotations

import mimetypes

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from ..jobs import JobCancelled, get_job_manager
from ..schemas import ExportOut, ExportRequest, JobOut
from ..storage import get_workspace, new_id, utcnow
from ..video import EXPORT_SUFFIX, ExportInputs, run_export
from .videos import require_ready_video

router = APIRouter(prefix="/api", tags=["exports"])


def _to_export_out(meta: dict) -> ExportOut:
    payload = {key: meta[key] for key in ExportOut.model_fields if key in meta}
    meta_obj = ExportOut(**payload)
    if meta.get("status") == "succeeded" and meta.get("filename"):
        meta_obj.download_url = (
            f"/api/videos/{meta['video_id']}/exports/{meta['id']}/download"
        )
    return meta_obj


def _export_body(ctx, *, video_id: str, export_id: str, kind: str,
                 session_id: str, options: dict) -> dict:
    """Runs on the job thread; mirrors status into the export record on every failure."""
    workspace = get_workspace()
    path = None
    try:
        ctx.check_cancelled()
        workspace.update_export_meta(video_id, export_id, status="running")
        meta = workspace.read_meta(video_id)
        inputs = ExportInputs(
            frames_dir=workspace.frames_dir(video_id),
            masks_dir=workspace.masks_dir(video_id, session_id),
            n_frames=int(meta.get("n_frames") or 0),
            fps=float(meta.get("fps") or 30.0),
            width=int(meta.get("frame_width") or 0),
            height=int(meta.get("frame_height") or 0),
            out_path=workspace.export_file_path(video_id, export_id, EXPORT_SUFFIX[kind]),
        )
        def progress(fraction, message=""):
            ctx.check_cancelled()
            ctx.progress(fraction, message)
        path = run_export(kind, inputs, progress, **options)
        ctx.check_cancelled()
        size = path.stat().st_size
        workspace.update_export_meta(
            video_id, export_id, status="succeeded",
            filename=path.name, size_bytes=size, completed_at=utcnow(),
        )
        return {
            "export_id": export_id,
            "kind": kind,
            "filename": path.name,
            "size_bytes": size,
            "download_url": f"/api/videos/{video_id}/exports/{export_id}/download",
        }
    except Exception as exc:  # noqa: BLE001 - record the reason, then let the job fail
        workspace.export_file_path(video_id, export_id, EXPORT_SUFFIX[kind]).unlink(missing_ok=True)
        workspace.update_export_meta(video_id, export_id,
                                     status="cancelled" if isinstance(exc, JobCancelled) else "failed",
                                     error=str(exc) or "export cancelled")
        raise


@router.post(
    "/videos/{video_id}/exports",
    response_model=JobOut,
    status_code=202,
    summary="Start an export job",
)
def create_export(video_id: str, payload: ExportRequest) -> JobOut:
    with get_workspace().operation_lock:
        require_ready_video(video_id)
        workspace = get_workspace()

        session_id = payload.session_id or workspace.latest_session_id(video_id)
        if session_id is None:
            raise HTTPException(status_code=409, detail="track an object before exporting")
        session = workspace.read_session_meta(video_id, session_id)
        if session.get("status", "succeeded") != "succeeded":
            raise HTTPException(status_code=409, detail="finish tracking successfully before exporting")

        options: dict = {}
        if payload.kind == "replace_bg":
            options = {"background": payload.background, "blur_radius": payload.blur_radius}

        manager = get_job_manager()
        job = manager.create("export", video_id=video_id)
        try:
            export_id = new_id()
            workspace.write_export_meta(
                video_id,
                export_id,
                {
                    "id": export_id,
                    "video_id": video_id,
                    "session_id": session_id,
                    "kind": payload.kind,
                    "status": "queued",
                    "filename": "",
                    "size_bytes": 0,
                    "created_at": utcnow(),
                    "error": None,
                    "options": options,
                    "job_id": job.id,
                },
            )

            def finish(job):
                if job.status in {"failed", "cancelled"}:
                    workspace.update_export_meta(video_id, export_id, status=job.status,
                                                 error=job.error or "export cancelled")
            manager.submit(
                job,
                lambda ctx: _export_body(
                    ctx, video_id=video_id, export_id=export_id,
                    kind=payload.kind, session_id=session_id, options=options,
                ),
                on_done=finish,
            )
            return JobOut(**job.to_dict())
        except BaseException:
            manager.cancel(job.id)
            raise


@router.get(
    "/videos/{video_id}/exports",
    response_model=list[ExportOut],
    summary="List exports",
)
def list_exports(video_id: str) -> list[ExportOut]:
    require_ready_video(video_id)
    return [_to_export_out(meta) for meta in get_workspace().list_exports(video_id)]


@router.get(
    "/videos/{video_id}/exports/{export_id}/download",
    summary="Download an export artifact",
)
def download_export(video_id: str, export_id: str) -> FileResponse:
    workspace = get_workspace()
    try:
        meta = workspace.read_export_meta(video_id, export_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    artifact = workspace.export_artifact(video_id, export_id)
    if artifact is None:
        raise HTTPException(
            status_code=409,
            detail=f"export {export_id} is {meta.get('status')!r} and has no artifact yet",
        )
    media_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
    return FileResponse(artifact, media_type=media_type, filename=artifact.name)


@router.delete(
    "/videos/{video_id}/exports/{export_id}",
    status_code=204,
    summary="Delete an export",
)
def delete_export(video_id: str, export_id: str) -> Response:
    with get_workspace().operation_lock:
        workspace = get_workspace()
        record = workspace.read_export_meta(video_id, export_id)
        if record.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="cancel or finish this export before deleting it")
        try:
            artifact = workspace.export_artifact(video_id, export_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if artifact is not None:
            artifact.unlink(missing_ok=True)
        workspace.export_meta_path(video_id, export_id).unlink(missing_ok=True)
        return Response(status_code=204)
