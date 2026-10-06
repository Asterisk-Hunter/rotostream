"""Drives a tracker over a video: prompt, propagate, persist masks and scores.

The step plan is built so the tracker never sees the future. With prompts at
frames ``p1 < p2 < ... < pk`` the order is::

    prompt(p1) -> propagate forward to p2-1 -> prompt(p2) -> ... -> propagate to the end
                                                            \\-> propagate backwards p1-1 .. 0

Each ``propagate`` step therefore only ever conditions on frames already visited in
its direction, which is exactly the causality rule in the plugin contract.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

from . import masks as mask_utils
from .jobs import JobContext, get_job_manager
from .models import registry
from .models.base import (
    ContractError,
    Direction,
    FrameResult,
    PromptSet,
    VideoObjectTracker,
    check_frame_result,
)
from .models.frames import DirectoryFrameSource
from .prompts import prompt_set_to_dict
from .quality import summarize_quality
from .settings import Settings
from .storage import Workspace, utcnow

logger = logging.getLogger(__name__)

Step = tuple[str, int] | tuple[str, int, Direction]


def _clamp(prompt: PromptSet, index: int) -> PromptSet:
    """Re-target a PromptSet at a valid frame index (clamped to the video length)."""
    return prompt if prompt.frame_index == index else replace(prompt, frame_index=index)


def build_plan(
    n_frames: int, prompt_indices: Sequence[int], bidirectional: bool
) -> list[Step]:
    """Leak-free ordering of prompt and propagate steps. See module docstring."""
    ordered = sorted({int(i) for i in prompt_indices if 0 <= int(i) < n_frames})
    if not ordered:
        raise ValueError("at least one prompt frame is required")

    steps: list[Step] = [("prompt", ordered[0])]
    cursor = ordered[0]

    for nxt in ordered[1:]:
        for index in range(cursor + 1, nxt):
            steps.append(("propagate", index, Direction.FORWARD))
        steps.append(("prompt", nxt))
        cursor = nxt

    for index in range(cursor + 1, n_frames):
        steps.append(("propagate", index, Direction.FORWARD))

    if bidirectional:
        for index in range(ordered[0] - 1, -1, -1):
            steps.append(("propagate", index, Direction.BACKWARD))

    return steps


@dataclass
class TrackingOutcome:
    n_tracked: int
    n_frames: int
    prompt_frames: list[int]
    mean_score: float
    absent_frames: list[int]
    elapsed_s: float
    scores: list[dict[str, Any]]
    memory: dict[str, Any]
    cancelled: bool = False


def has_positive_seed(prompt: PromptSet) -> bool:
    """True when a prompt can define an object: a box or a foreground click."""
    return prompt.box is not None or any(point.positive for point in prompt.points)


def frame_source(workspace: Workspace, video_id: str, meta: dict[str, Any]) -> DirectoryFrameSource:
    return DirectoryFrameSource(
        workspace.frames_dir(video_id),
        fps=float(meta.get("fps") or 30.0),
    )


def _instantiate(model_key: str, settings: Settings, checkpoint: str | None):
    """Create and load a tracker, turning contract violations into clear errors."""
    try:
        tracker = registry.create(model_key)
    except KeyError as exc:
        raise ContractError(str(exc)) from exc
    tracker.load(device=settings.device, checkpoint=checkpoint)
    if not hasattr(tracker, "set_video"):
        raise ContractError(f"{model_key} does not implement set_video()")
    return tracker


# ------------------------------------------------------------------ preview cache
# Clicking in the studio fires one preview per interaction. Building a tracker (and
# loading weights) on every click would be unusable for a real model, so keep one
# warm instance per (video, model, checkpoint) and reuse it. Access is serialised:
# a single GPU cannot serve two overlapping forwards without thrashing VRAM.
_preview_lock = threading.Lock()
_preview_cache: OrderedDict[tuple, tuple[VideoObjectTracker, str]] = OrderedDict()


def preview_mask(
    *,
    workspace: Workspace,
    settings: Settings,
    video_id: str,
    prompt: PromptSet,
    model_key: str,
    checkpoint: str | None = None,
) -> np.ndarray:
    """Mask for a single prompted frame - powers instant click feedback in the UI."""
    meta = workspace.read_meta(video_id)
    frames = frame_source(workspace, video_id, meta)
    frame_index = min(max(prompt.frame_index, 0), frames.n_frames - 1)
    key = (video_id, model_key, checkpoint or "", settings.device)

    with get_job_manager().compute_slot(), _preview_lock:
        cached = _preview_cache.get(key)
        if cached is None:
            tracker = _instantiate(model_key, settings, checkpoint)
            tracker.set_video(frames)
            tracker.warmup()
        else:
            tracker, attached = cached
            if attached != video_id:
                tracker.set_video(frames)
                tracker.warmup()
        _preview_cache[key] = (tracker, video_id)
        _preview_cache.move_to_end(key)
        while len(_preview_cache) > settings.preview_cache_size:
            _preview_cache.popitem(last=False)

        tracker.reset()
        result = tracker.add_prompt(_clamp(prompt, frame_index))
        check_frame_result(result, frames.shape)
        return result.mask


def clear_preview_cache(video_id: str | None = None) -> None:
    """Drop warm preview trackers (tests, or after a model checkpoint changes)."""
    with _preview_lock:
        if video_id is None:
            _preview_cache.clear()
        else:
            for key in list(_preview_cache):
                if key[0] == video_id:
                    del _preview_cache[key]


def run_tracking(
    ctx: JobContext,
    *,
    workspace: Workspace,
    settings: Settings,
    video_id: str,
    session_id: str,
    prompts: Sequence[PromptSet],
    model_key: str,
    bidirectional: bool = True,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    ctx.check_cancelled()
    # The worker already holds the shared compute gate. Release cached preview
    # model references before loading this session's tracker so VRAM holds one
    # model instance, including when the preview used a different tracker.
    clear_preview_cache()
    meta = workspace.read_meta(video_id)
    frames = frame_source(workspace, video_id, meta)
    height, width = frames.shape

    prompts_by_frame: dict[int, PromptSet] = {}
    for prompt in prompts:
        index = min(max(int(prompt.frame_index), 0), frames.n_frames - 1)
        prompts_by_frame[index] = _clamp(prompt, index)
    # A prompt with no box and no foreground click is a deliberate "background
    # here" answer. It yields no mask on that frame by design, which must not be
    # reported to the studio as lost tracking.
    background_only = sorted(
        index for index, prompt in prompts_by_frame.items() if not has_positive_seed(prompt)
    )

    steps = build_plan(frames.n_frames, list(prompts_by_frame), bidirectional)
    ctx.note(f"loading {model_key}")
    tracker = _instantiate(model_key, settings, checkpoint)
    tracker.set_video(frames)
    tracker.warmup()

    scores: list[dict[str, Any]] = []
    absent: list[int] = []
    masks_dir = workspace.masks_dir(video_id, session_id)
    total = len(steps)

    for position, step in enumerate(steps):
        ctx.check_cancelled()
        frame_index = int(step[1])
        if step[0] == "prompt":
            result: FrameResult = tracker.add_prompt(prompts_by_frame[frame_index])
            prompted = True
        else:
            direction = step[2]  # type: ignore[misc]
            result = tracker.propagate(frame_index, direction)
            prompted = False

        try:
            check_frame_result(result, frames.shape)
        except ContractError as exc:
            hint = ""
            if prompted and frame_index in set(background_only):
                hint = (
                    " This frame has background clicks only, and "
                    f"{model_key} needs a foreground click or a box on every prompted "
                    "frame: add one on this frame, or remove this frame's prompts."
                )
            raise ContractError(f"[{model_key}] frame {frame_index}: {exc}{hint}") from exc

        mask_utils.save_mask(masks_dir / f"{frame_index:06d}.png", result.mask)
        scores.append(
            {
                "frame_index": frame_index,
                "score": float(result.score),
                "object_present": bool(result.object_present),
                "prompted": prompted,
            }
        )
        if not result.object_present and frame_index not in background_only:
            absent.append(frame_index)

        ctx.progress(
            (position + 1) / total,
            f"{'prompt' if prompted else 'tracking'} frame {frame_index + 1}/{frames.n_frames}",
        )

    scores.sort(key=lambda record: record["frame_index"])
    elapsed = time.time() - started
    mean_score = float(np.mean([s["score"] for s in scores])) if scores else 0.0
    quality = summarize_quality(
        scores,
        n_frames=frames.n_frames,
        prompt_frames=list(prompts_by_frame),
        background_only_frames=background_only,
    )

    try:
        memory = tracker.memory_state() or {}
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail a job
        logger.warning("memory_state() raised: %s", exc)
        memory = {"error": str(exc)}

    outcome = TrackingOutcome(
        n_tracked=len(scores),
        n_frames=frames.n_frames,
        prompt_frames=sorted(prompts_by_frame),
        mean_score=mean_score,
        absent_frames=sorted(absent),
        elapsed_s=elapsed,
        scores=scores,
        memory=memory,
    )

    session_meta = {
        "id": session_id,
        "video_id": video_id,
        "model": model_key,
        "created_at": utcnow(),
        "n_tracked": outcome.n_tracked,
        "n_frames": outcome.n_frames,
        "prompt_frames": outcome.prompt_frames,
        "mean_score": outcome.mean_score,
        "absent_frames": outcome.absent_frames,
        "elapsed_s": outcome.elapsed_s,
        "cancelled": False,
        "scores": outcome.scores,
        "memory": outcome.memory,
        "checkpoint": checkpoint,
        "bidirectional": bidirectional,
        "status": "succeeded",
        # Self-describing sessions: the exact prompts that produced these masks, so
        # a reload can restore them and a later run cannot silently drop them.
        "prompts": [prompt_set_to_dict(prompts_by_frame[index]) for index in sorted(prompts_by_frame)],
        "quality": quality,
    }
    ctx.check_cancelled()
    workspace.write_session_meta(video_id, session_id, session_meta)

    return {
        "session_id": session_id,
        "video_id": video_id,
        "model": model_key,
        "n_tracked": outcome.n_tracked,
        "n_frames": outcome.n_frames,
        "mean_score": round(outcome.mean_score, 4),
        "prompt_frames": outcome.prompt_frames,
        "absent_frames": outcome.absent_frames,
        "elapsed_s": round(outcome.elapsed_s, 2),
        "memory": outcome.memory,
        "quality": quality,
    }
