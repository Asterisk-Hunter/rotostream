"""Synthetic toy sequences with ground truth, for contract tests.

These exist to catch directionality and indexing bugs — memory attending to the
*future* instead of the past, off-by-one frame ordering — which are far faster to
find here than on real footage, where everything looks "sort of okay".

Ground truth masks are empty for frames where the object is fully occluded, which
is what the occlusion head has to report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.frames import ArrayFrameSource

BACKGROUND: tuple[int, int, int] = (28, 30, 38)
OBJECT: tuple[int, int, int] = (232, 74, 62)
OCCLUDER: tuple[int, int, int] = (198, 200, 206)

DEFAULT_SIZE = (96, 128)


@dataclass
class ToySequence:
    frames: np.ndarray  # (T, H, W, 3) uint8
    masks: np.ndarray  # (T, H, W) bool ground truth
    fps: float = 10.0

    def source(self) -> ArrayFrameSource:
        return ArrayFrameSource(self.frames, fps=self.fps)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.frames.shape[1]), int(self.frames.shape[2]))

    @property
    def centre_y(self) -> int:
        return self.size[0] // 2


def _canvas(n_frames: int, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = size
    frames = np.empty((n_frames, height, width, 3), np.uint8)
    frames[:] = BACKGROUND
    return frames, np.zeros((n_frames, height, width), bool)


def _paint(frames, masks, t, centre, half, colour=OBJECT) -> None:
    height, width = frames.shape[1:3]
    cx, cy = centre
    x0, x1 = max(0, cx - half), min(width, cx + half)
    y0, y1 = max(0, cy - half), min(height, cy + half)
    frames[t, y0:y1, x0:x1] = colour
    masks[t, y0:y1, x0:x1] = True


def sliding_square(
    n_frames: int = 8,
    size: tuple[int, int] = DEFAULT_SIZE,
    half: int = 10,
    *,
    x_start: int = 20,
    x_end: int | None = None,
    fps: float = 10.0,
) -> ToySequence:
    """A solid square gliding horizontally. The sanity baseline."""
    frames, masks = _canvas(n_frames, size)
    height, width = size
    x_end = (width - half - 4) if x_end is None else x_end
    for t in range(n_frames):
        fraction = 0.0 if n_frames == 1 else t / (n_frames - 1)
        centre_x = int(round(x_start + (x_end - x_start) * fraction))
        _paint(frames, masks, t, (centre_x, height // 2), half)
    return ToySequence(frames, masks, fps)


def teleporting_square(
    n_frames: int = 8,
    size: tuple[int, int] = DEFAULT_SIZE,
    half: int = 10,
    *,
    switch_at: int = 4,
    x_before: int = 20,
    x_after: int = 100,
    fps: float = 10.0,
) -> ToySequence:
    """Square sits at ``x_before``, then jumps to ``x_after`` at ``switch_at``.

    Used for future-leakage tests: two variants that share ``x_before`` but differ
    in ``x_after`` (or vice versa) must not influence predictions on the shared side.
    """
    frames, masks = _canvas(n_frames, size)
    height, _ = size
    for t in range(n_frames):
        centre_x = x_before if t < switch_at else x_after
        _paint(frames, masks, t, (centre_x, height // 2), half)
    return ToySequence(frames, masks, fps)


def occluded_square(
    n_frames: int = 12,
    size: tuple[int, int] = DEFAULT_SIZE,
    half: int = 10,
    *,
    bar_half: int = 8,
    fps: float = 10.0,
) -> ToySequence:
    """A static bar stands in the middle; the square slides right and passes behind it.

    While the square overlaps the bar it is fully hidden, so the ground-truth mask is
    empty and ``object_present`` must be False. After that it re-emerges, which is the
    re-entry case a memory-free tracker cannot handle.
    """
    frames, masks = _canvas(n_frames, size)
    height, width = size
    bar_x = width // 2

    for t in range(n_frames):
        fraction = t / max(1, n_frames - 1)
        centre_x = int(round(20 + (width - half - 4 - 20) * fraction))

        frames[t, :, bar_x - bar_half : bar_x + bar_half] = OCCLUDER
        hidden = abs(centre_x - bar_x) < (bar_half + half)
        if not hidden:
            _paint(frames, masks, t, (centre_x, height // 2), half)

    return ToySequence(frames, masks, fps)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union; two empty masks count as a perfect match."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)
