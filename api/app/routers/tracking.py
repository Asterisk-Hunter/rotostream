"""Interactive prompting, tracking jobs and job progress streaming."""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .. import masks as mask_utils
from ..jobs import get_job_manager
from ..models import registry
from ..models.base import ContractError
from ..pipeline import preview_mask, run_tracking
from ..prompts import PromptError, to_prompt_set
from ..schemas import JobOut, PreviewRequest, TrackRequest
from ..settings import get_settings
from ..storage import get_workspace, utcnow
from ..video import read_frame
from .videos import require_ready_video

router = APIRouter(prefix="/api", tags=["tracking"])

_MODEL_HELP = "Choose an available tracker from the model selector."


def assert_model_usable(model_key: str) -> None:
    """Fail fast with 404/409 instead of letting a background job die on the worker."""
    infos = {info.name: info for info in registry.available()}
    info = infos.get(model_key)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown tracker {model_key!r}; known: {', '.join(sorted(infos))}",
        )
    if not info.implemented:
        reason = info.error or "not implemented yet"
        hint = info.checkpoint_hint or _MODEL_HELP
        raise HTTPException(
            status_code=409,
            detail=f"tracker {model_key!r} is not usable ({reason}). {hint}",
        )


# ---------------------------------------------------------------------- preview
@router.post(
    "/videos/{video_id}/preview",
    summary="Mask for a single prompted frame",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def preview(video_id: str, payload: PreviewRequest) -> Response:
    """Run the tracker on one frame so clicking in the studio gives instant feedback.

    Returns the overlay PNG directly and puts mask stats in headers, which keeps
    the interactive path to a single request per click.
    """
    with get_workspace().operation_lock:
        meta = require_ready_video(video_id)
        settings = get_settings()
        workspace = get_workspace()
        model_key = payload.model or settings.default_model
        assert_model_usable(model_key)

        try:
            prompt = to_prompt_set(payload.prompt)
        except PromptError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        n_frames = int(meta.get("n_frames") or 0)
        if prompt.frame_index >= n_frames:
            raise HTTPException(
                status_code=422, detail=f"frame_index {prompt.frame_index} is beyond {n_frames} frames"
            )

        try:
            mask = preview_mask(
                workspace=workspace,
                settings=settings,
                video_id=video_id,
                prompt=prompt,
                model_key=model_key,
                checkpoint=payload.checkpoint,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ContractError as exc:
            raise HTTPException(status_code=422, detail=f"tracker contract violation: {exc}") from exc

        frame = read_frame(workspace.frames_dir(video_id), prompt.frame_index)
        png = mask_utils.overlay_png_bytes(frame, mask)
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Mask-Area": str(int(mask.sum())),
                "X-Mask-Ratio": f"{float(mask.mean()):.6f}",
            },
        )


# --------------------------------------------------------------------- tracking
@router.post(
    "/videos/{video_id}/track",
    response_model=JobOut,
    status_code=202,
    summary="Start a tracking job",
)
def start_tracking(video_id: str, payload: TrackRequest) -> JobOut:
    with get_workspace().operation_lock:
        meta = require_ready_video(video_id)
        settings = get_settings()
        workspace = get_workspace()
        model_key = payload.model or settings.default_model
        assert_model_usable(model_key)

        n_frames = int(meta.get("n_frames") or 0)
        try:
            prompt_sets = [to_prompt_set(prompt) for prompt in payload.prompts]
        except PromptError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for prompt in prompt_sets:
            if prompt.frame_index >= n_frames:
                raise HTTPException(
                    status_code=422,
                    detail=f"prompt frame_index {prompt.frame_index} is beyond {n_frames} frames",
                )
        if len({prompt.frame_index for prompt in prompt_sets}) != len(prompt_sets):
            raise HTTPException(status_code=422, detail="combine prompts for each frame into one prompt")

        if payload.session_id:
            raise HTTPException(status_code=409, detail="sessions are immutable; start a new tracking session")

        manager = get_job_manager()
        job = manager.create("track", video_id=video_id)
        try:
            session_id = workspace.create_session(video_id)
            workspace.write_session_meta(video_id, session_id, {
                "id": session_id, "video_id": video_id, "model": model_key,
                "created_at": utcnow(), "status": "queued", "n_frames": n_frames,
                "prompt_frames": sorted(prompt.frame_index for prompt in prompt_sets),
            })

            def finish(job):
                if job.status != "succeeded":
                    record = workspace.read_session_meta(video_id, session_id)
                    record.update(status=job.status, cancelled=job.status == "cancelled", error=job.error)
                    workspace.write_session_meta(video_id, session_id, record)

            def track(ctx):
                record = workspace.read_session_meta(video_id, session_id)
                record["status"] = "running"
                workspace.write_session_meta(video_id, session_id, record)
                return run_tracking(
                    ctx, workspace=workspace, settings=settings, video_id=video_id,
                    session_id=session_id, prompts=prompt_sets, model_key=model_key,
                    bidirectional=payload.bidirectional, checkpoint=payload.checkpoint,
                )

            manager.submit(
                job, track, on_done=finish,
            )
            return JobOut(**job.to_dict())
        except BaseException:
            manager.cancel(job.id)
            raise


# ------------------------------------------------------------------------- jobs
@router.get("/jobs", response_model=list[JobOut], summary="List jobs")
def list_jobs(video_id: str | None = None) -> list[JobOut]:
    return [JobOut(**job.to_dict()) for job in get_job_manager().list(video_id=video_id)]


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Job status")
def get_job(job_id: str) -> JobOut:
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return JobOut(**job.to_dict())


@router.post("/jobs/{job_id}/cancel", response_model=JobOut, summary="Cancel a job")
def cancel_job(job_id: str) -> JobOut:
    manager = get_job_manager()
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    manager.cancel(job_id)
    job = manager.get(job_id)
    assert job is not None
    return JobOut(**job.to_dict())


@router.get("/jobs/{job_id}/events", summary="Server-sent job progress")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    manager = get_job_manager()
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    async def stream():
        last: str | None = None
        heartbeat = time.monotonic()
        while True:
            job = manager.get(job_id)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'detail': 'job disappeared'})}\n\n"
                return
            encoded = json.dumps(job.to_dict())
            if encoded != last:
                yield f"data: {encoded}\n\n"
                last = encoded
                heartbeat = time.monotonic()
            elif time.monotonic() - heartbeat >= 15:
                yield ": keepalive\n\n"
                heartbeat = time.monotonic()
            if job.done:
                yield "event: done\ndata: {}\n\n"
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.15)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
