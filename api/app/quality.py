"""Honest review rules for a tracking run.

Lives outside both the pipeline and the response schema so an old session can be
described with the same numbers a new run would produce. Both callers must be able
to say "this clip is not finished": the studio is not allowed to imply a complete
mask when frames are missing or weak.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: A propagated frame whose score falls below this is reported as low quality.
#: It is a *review* threshold, not a contract rule: 0.5 is where the mask decoder's
#: predicted IoU stops being a reliable statement that the mask is on the object.
LOW_CONFIDENCE_THRESHOLD = 0.5

#: The share of low-confidence frames that still counts as a sound run.
SOUND_LOW_CONFIDENCE_SHARE = 0.05


def summarize_quality(
    scores: Sequence[Mapping[str, Any]],
    *,
    n_frames: int,
    prompt_frames: Sequence[int],
    background_only_frames: Sequence[int] = (),
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Derive the honest review summary for a finished run.

    Three things are deliberately distinct, because the studio must never imply a
    finished mask when parts of the clip are missing or weak:

    * ``absent_frames`` -- propagated frames where the tracker reported no object.
      A frame the user marked as background *on purpose* is excluded: that answer is
      their instruction, not lost tracking.
    * ``low_confidence_frames`` -- frames with a mask but a score below ``threshold``.
      Prompted frames are excluded: a prompt is ground truth from the user, so its
      score is a property of the model's self-assessment, not of the mask.
    * ``problem_frames`` -- the union, sorted, which is what the review controls jump
      between and what the export warning counts.
    """
    background_only = {int(index) for index in background_only_frames}
    prompted = {int(index) for index in prompt_frames}

    absent = sorted(
        int(record["frame_index"])
        for record in scores
        if not record.get("object_present", True) and int(record["frame_index"]) not in background_only
    )
    low = sorted(
        int(record["frame_index"])
        for record in scores
        if record.get("object_present", True)
        and float(record.get("score", 0.0)) < threshold
        and int(record["frame_index"]) not in prompted
    )
    present = [record for record in scores if record.get("object_present", True)]
    values = [float(record.get("score", 0.0)) for record in present]
    problem = sorted(set(absent) | set(low))
    coverage = (len(present) / n_frames) if n_frames else 0.0
    sound = not absent and len(low) <= max(0, int(SOUND_LOW_CONFIDENCE_SHARE * max(1, n_frames)))

    return {
        "threshold": threshold,
        "n_frames": int(n_frames),
        "n_masked": len(present),
        "coverage": round(coverage, 4),
        "absent_frames": absent,
        "low_confidence_frames": low,
        "background_only_frames": sorted(background_only),
        "problem_frames": problem,
        "mean_score": round(float(np.mean(values)), 4) if values else 0.0,
        "min_score": round(float(np.min(values)), 4) if values else 0.0,
        "sound": bool(sound),
    }


def session_quality(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """The stored review summary, or one derived from the session's own scores.

    Sessions written before the summary existed still carry their per-frame scores
    and their recorded absent frames, so derive from those instead of falling back to
    defaults. Reporting "coverage 100%, sound" for a clip the tracker lost half of is
    exactly the claim the studio must not make.
    """
    meta = meta or {}
    stored = meta.get("quality") or {}
    if stored.get("n_frames"):
        return dict(stored)
    scores = [record for record in (meta.get("scores") or []) if isinstance(record, Mapping)]
    n_frames = int(meta.get("n_frames") or len(scores))
    summary = summarize_quality(
        scores,
        n_frames=n_frames,
        prompt_frames=list(meta.get("prompt_frames") or []),
        background_only_frames=list(meta.get("background_only_frames") or ()),
    )
    recorded = sorted({int(index) for index in (meta.get("absent_frames") or [])})
    if recorded and recorded != summary["absent_frames"]:
        # An older run recorded its own absent list; trust the run over a re-derivation
        # (it saw the background-only prompt frames this meta no longer stores).
        summary["absent_frames"] = recorded
        summary["n_masked"] = max(0, n_frames - len(recorded))
        summary["coverage"] = round(summary["n_masked"] / n_frames, 4) if n_frames else 0.0
        summary["problem_frames"] = sorted(set(recorded) | set(summary["low_confidence_frames"]))
        summary["sound"] = False
    return summary
