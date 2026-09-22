"""Deterministic toy sequences with exact ground truth.

Real footage hides indexing bugs: a tracker that attends to the wrong frames, or
one that is off by one in the memory queue, still produces masks that look
"sort of okay", and you lose a day to it. These sequences are built so the right
answer is known analytically, which makes those bugs obvious in seconds:

* ``linear``      - constant-velocity square, no occlusion. Baseline plumbing.
* ``occlusion``   - the square passes behind an opaque bar and is *fully* hidden
                    for several frames, then reappears. This is the object
                    presence / occlusion-head test, and the one naive trackers
                    fail: they either freeze the last mask or drift.
* ``reentry``     - the square leaves the frame entirely and comes back, so the
                    object is genuinely absent from the image, not just covered.
* ``distractor``  - two identical squares. The prompt selects one; following the
                    other is the classic memory-attention failure.
* ``color_shift`` - the object changes colour mid-sequence, so a tracker that
                    matches appearance rather than identity drifts.

Every builder returns a :class:`ToySequence` with ``frames`` (T, H, W, 3) uint8,
per-object ``masks`` (T, N, H, W) bool, and a one-click prompt per object taken
from the centroid of its first visible frame. ``prompt_for`` gives you a
:class:`~app.models.base.PromptSet`, so a sequence drops straight into any
tracker implementing the API contract.

Ground truth for partially occluded frames is the *visible* extent of the object,
matching how DAVIS annotates occlusion, so a tracker that paints the full square
through the bar is correctly penalised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .sequences import VideoSequence

#: Synthetic clips are ordinary :class:`VideoSequence` instances; the alias just
#: reads better at call sites and in ``isinstance`` checks.
ToySequence = VideoSequence

__all__ = [
    "ToySequence",
    "linear",
    "occlusion",
    "reentry",
    "distractor",
    "color_shift",
    "build",
    "available",
    "BUILDERS",
]

RGB = tuple[int, int, int]

#: Object colours, chosen to be well separated in RGB.
OBJECT_COLOURS: tuple[RGB, ...] = ((220, 60, 50), (60, 120, 220), (240, 200, 60))
OCCLUDER_COLOUR: RGB = (25, 28, 34)


def _background(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A static, textured background so no tracker can win by predicting nothing.

    Texture matters: against a flat background, "everything that moved" is a
    trivially correct answer and these sequences stop testing memory.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 30 + (xx / max(width - 1, 1) * 70).astype(np.uint8)
    frame[..., 1] = 40 + (yy / max(height - 1, 1) * 55).astype(np.uint8)
    frame[..., 2] = 75
    noise = rng.integers(0, 14, size=(height, width, 1), dtype=np.uint8)
    return np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)


def _clip_bounds(
    x: float, y: float, width: float, height: float, frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(frame_width, int(round(x + width)))
    y1 = min(frame_height, int(round(y + height)))
    return x0, y0, x1, y1


def _paint(frame: np.ndarray, box: tuple[int, int, int, int], colour: RGB) -> None:
    x0, y0, x1, y1 = box
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = colour


def _mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    if x0 < x1 and y0 < y1:
        out[y0:y1, x0:x1] = True
    return out


# ---------------------------------------------------------------------- builders
def linear(
    n_frames: int = 24,
    height: int = 96,
    width: int = 128,
    size: int = 16,
    speed: float = 3.0,
    fps: float = 10.0,
    seed: int = 0,
) -> ToySequence:
    """A square crossing the frame at constant velocity. The plumbing baseline."""
    frames, masks = [], []
    for t in range(n_frames):
        frame = _background(height, width, seed)
        x = 6 + speed * t
        y = (height - size) / 2
        box = _clip_bounds(x, y, size, size, width, height)
        _paint(frame, box, OBJECT_COLOURS[0])
        masks.append(_mask((height, width), box)[None])
        frames.append(frame)
    return ToySequence(
        name="linear",
        frames=np.stack(frames),
        masks=np.stack(masks),
        fps=fps,
        meta={"speed": speed, "size": size},
    )


def occlusion(
    n_frames: int = 30,
    height: int = 96,
    width: int = 160,
    size: int = 18,
    speed: float = 4.0,
    bar_width: int = 40,
    fps: float = 10.0,
    seed: int = 0,
) -> ToySequence:
    """A square passes behind an opaque bar and is fully hidden for several frames.

    The bar is drawn *in front* of the square, and ground truth is the visible
    extent, so the frames where the square is fully covered have an empty mask and
    ``visible=False``. A tracker without an object-presence head typically keeps
    emitting the last mask there, which shows up as a cliff in the per-frame J.
    """
    bar_x = int(width * 0.5)
    frames, masks = [], []
    for t in range(n_frames):
        frame = _background(height, width, seed)
        x = 8 + speed * t
        y = (height - size) / 2
        square = _clip_bounds(x, y, size, size, width, height)
        bar = _clip_bounds(bar_x, 0, bar_width, height, width, height)

        # Visible extent = square minus whatever the bar covers.
        square_mask = _mask((height, width), square)
        bar_mask = _mask((height, width), bar)
        visible_mask = square_mask & ~bar_mask

        # Draw the square first, then the bar over it.
        _paint(frame, square, OBJECT_COLOURS[0])
        _paint(frame, bar, OCCLUDER_COLOUR)

        frames.append(frame)
        masks.append(visible_mask[None])
    return ToySequence(
        name="occlusion",
        frames=np.stack(frames),
        masks=np.stack(masks),
        fps=fps,
        meta={"bar_x": bar_x, "bar_width": bar_width, "size": size},
    )


def reentry(
    n_frames: int = 34,
    height: int = 96,
    width: int = 128,
    size: int = 16,
    speed: float = 8.0,
    gap: int = 4,
    fps: float = 10.0,
    seed: int = 0,
) -> ToySequence:
    """The square leaves the frame entirely, is absent for a few frames, then returns.

    Harder than occlusion: nothing covers the object, so for the gap frames there
    is genuinely no object signal anywhere in the image. Only a tracker that can
    report "not present" and then re-acquire from memory scores well; one that
    keeps emitting the last mask scores zero on those frames.
    """
    y = (height - size) / 2
    x_start = 4.0
    # First frame at which the square is completely off the right edge.
    exit_frame = int(np.ceil((width - x_start) / speed))
    # Then it waits off-screen and re-enters from the left.
    reentry_frame = exit_frame + gap

    frames, masks = [], []
    for t in range(n_frames):
        frame = _background(height, width, seed)
        if t < exit_frame:
            x = x_start + speed * t
        elif t < reentry_frame:
            x = width + speed * (t - exit_frame)  # entirely off the right edge
        else:
            x = -(size + 6) + speed * (t - reentry_frame)  # re-enters from the left

        box = _clip_bounds(x, y, size, size, width, height)
        _paint(frame, box, OBJECT_COLOURS[0])
        frames.append(frame)
        masks.append(_mask((height, width), box)[None])
    return ToySequence(
        name="reentry",
        frames=np.stack(frames),
        masks=np.stack(masks),
        fps=fps,
        meta={"speed": speed, "size": size, "gap": gap, "exit_frame": exit_frame},
    )


def distractor(
    n_frames: int = 24,
    height: int = 96,
    width: int = 160,
    size: int = 16,
    fps: float = 10.0,
    seed: int = 0,
) -> ToySequence:
    """Two identical squares moving in opposite directions.

    Object 0 starts on the left and moves right; object 1 starts on the right and
    moves left. They are the same size and colour, so only memory can keep them
    apart - following the wrong one scores near zero on object 0 and looks
    plausible at a glance.
    """
    frames, masks = [], []
    y_top = height * 0.25 - size / 2
    y_bottom = height * 0.75 - size / 2
    speed = 4.0
    for t in range(n_frames):
        frame = _background(height, width, seed)
        box_a = _clip_bounds(6 + speed * t, y_top, size, size, width, height)
        box_b = _clip_bounds(width - 6 - size - speed * t, y_bottom, size, size, width, height)
        _paint(frame, box_a, OBJECT_COLOURS[0])
        _paint(frame, box_b, OBJECT_COLOURS[0])  # identical colour, on purpose
        frames.append(frame)
        masks.append(
            np.stack([_mask((height, width), box_a), _mask((height, width), box_b)])
        )
    return ToySequence(
        name="distractor",
        frames=np.stack(frames),
        masks=np.stack(masks),
        fps=fps,
        meta={"speed": speed, "size": size},
    )


def color_shift(
    n_frames: int = 24,
    height: int = 96,
    width: int = 128,
    size: int = 18,
    speed: float = 3.0,
    fps: float = 10.0,
    seed: int = 0,
) -> ToySequence:
    """The object changes colour mid-clip, so appearance matching alone drifts."""
    frames, masks = [], []
    for t in range(n_frames):
        frame = _background(height, width, seed)
        x = 6 + speed * t
        y = (height - size) / 2
        box = _clip_bounds(x, y, size, size, width, height)

        blend = min(1.0, max(0.0, (t - n_frames / 2) / (n_frames / 4)))
        start, end = OBJECT_COLOURS[0], OBJECT_COLOURS[1]
        colour = tuple(
            int(round(a + (b - a) * blend)) for a, b in zip(start, end)
        )
        _paint(frame, box, colour)
        frames.append(frame)
        masks.append(_mask((height, width), box)[None])
    return ToySequence(
        name="color_shift",
        frames=np.stack(frames),
        masks=np.stack(masks),
        fps=fps,
        meta={"speed": speed, "size": size},
    )


BUILDERS: dict[str, Callable[..., ToySequence]] = {
    "linear": linear,
    "occlusion": occlusion,
    "reentry": reentry,
    "distractor": distractor,
    "color_shift": color_shift,
}


def available() -> list[str]:
    """Names of every synthetic scenario."""
    return sorted(BUILDERS)


def build(name: str, **kwargs: Any) -> ToySequence:
    """Build a scenario by name. Unknown names raise with the valid list."""
    try:
        builder = BUILDERS[name]
    except KeyError:
        raise KeyError(f"unknown synthetic sequence {name!r}; try one of {available()}") from None
    return builder(**kwargs)
