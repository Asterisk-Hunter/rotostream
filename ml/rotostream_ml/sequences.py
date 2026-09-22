"""The one sequence type the harness passes around.

Both the synthetic toy clips and DAVIS load into a :class:`VideoSequence`, so
evaluation, causality checking and training all read the same object. Frames are
either in memory (synthetic) or on disk (DAVIS), and ``frame_source`` hands the
tracker whichever the underlying ``FrameSource`` implementation expects - the
same lazy, random-access interface the API uses, so a tracker behaves identically
under evaluation and in the app.

Ground truth is always ``(T, N, H, W)`` bool: ``T`` frames, ``N`` objects. One
object is tracked per session (that is what a prompt selects), so consumers index
an object with :meth:`VideoSequence.object_masks`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.models.base import PointPrompt, PromptSet
from app.models.frames import ArrayFrameSource, DirectoryFrameSource, FrameSource

__all__ = ["VideoSequence", "interior_point"]


def interior_point(mask: np.ndarray) -> tuple[float, float] | None:
    """A point safely inside ``mask``, or ``None`` if the mask is empty.

    Uses the centre of the largest inscribed disk (the argmax of the distance
    transform) rather than the centroid: for a crescent, a person, or any shape
    with a concave side, the centroid can land outside the object, which would
    simulate a click the user never made. Ties break by row-major order, so the
    prompt is deterministic across runs.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return None
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(distance)), mask.shape)
    return float(x), float(y)


@dataclass
class VideoSequence:
    """A video with per-object ground truth.

    Exactly one of ``frames`` / ``frame_dir`` must be given. ``masks`` is always
    materialised: it is one bit per pixel, and reading it lazily buys nothing but
    complexity in every consumer.
    """

    name: str
    masks: np.ndarray  # (T, N, H, W) bool
    frames: np.ndarray | None = None  # (T, H, W, 3) uint8, when held in memory
    frame_dir: Path | None = None  # or frames on disk as JPEG/PNG
    fps: float = 24.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames is None and self.frame_dir is None:
            raise ValueError(f"{self.name}: one of frames / frame_dir is required")
        if self.frames is not None and self.frame_dir is not None:
            raise ValueError(f"{self.name}: pass frames or frame_dir, not both")

        self.masks = np.ascontiguousarray(self.masks, dtype=bool)
        if self.masks.ndim != 4:
            raise ValueError(f"{self.name}: masks must be (T, N, H, W), got {self.masks.shape}")

        if self.frames is not None:
            if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
                raise ValueError(
                    f"{self.name}: frames must be (T, H, W, 3), got {self.frames.shape}"
                )
            if self.frames.shape[0] != self.masks.shape[0]:
                raise ValueError(f"{self.name}: frames and masks disagree on T")
            if self.frames.shape[1:3] != self.masks.shape[2:4]:
                raise ValueError(f"{self.name}: frames and masks disagree on H/W")
            self.frames = np.ascontiguousarray(self.frames, dtype=np.uint8)
        elif self.frame_dir is not None:
            self.frame_dir = Path(self.frame_dir)
            if not self.frame_dir.is_dir():
                raise FileNotFoundError(f"{self.name}: no such frame directory {self.frame_dir}")

    # ------------------------------------------------------------------ shape
    @property
    def n_frames(self) -> int:
        return int(self.masks.shape[0])

    @property
    def n_objects(self) -> int:
        return int(self.masks.shape[1])

    @property
    def height(self) -> int:
        return int(self.masks.shape[2])

    @property
    def width(self) -> int:
        return int(self.masks.shape[3])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def __len__(self) -> int:
        return self.n_frames

    def __repr__(self) -> str:
        source = "memory" if self.frames is not None else f"{self.frame_dir}"
        return (
            f"VideoSequence({self.name!r}, {self.n_frames} frames, "
            f"{self.n_objects} object(s), {self.width}x{self.height}, {source})"
        )

    # ------------------------------------------------------------- ground truth
    def object_masks(self, object_index: int = 0) -> np.ndarray:
        """(T, H, W) ground truth for one object."""
        return self.masks[:, object_index]

    def visible(self, object_index: int = 0) -> np.ndarray:
        """(T,) whether the object is on screen in each frame."""
        return self.object_masks(object_index).reshape(self.n_frames, -1).any(axis=1)

    def first_visible_frame(self, object_index: int = 0) -> int:
        indices = np.nonzero(self.visible(object_index))[0]
        if len(indices) == 0:
            raise ValueError(f"{self.name}: object {object_index} is never visible")
        return int(indices[0])

    def present_flag_per_frame(self, object_index: int = 0) -> list[bool]:
        """Ground truth for the object-presence head, per frame."""
        return [bool(flag) for flag in self.visible(object_index)]

    # ---------------------------------------------------------------- prompts
    def prompt_for(self, object_index: int = 0, frame_index: int | None = None) -> PromptSet:
        """A single positive click inside the object, as a user would place it.

        Defaults to the first frame the object is visible. Raises if the object is
        not visible there rather than returning a click on the background.
        """
        index = self.first_visible_frame(object_index) if frame_index is None else frame_index
        point = interior_point(self.object_masks(object_index)[index])
        if point is None:
            raise ValueError(
                f"{self.name}: object {object_index} is not visible in frame {index}"
            )
        return PromptSet(
            frame_index=index,
            points=(PointPrompt(x=point[0], y=point[1], positive=True),),
        )

    # ----------------------------------------------------------------- frames
    def frame_source(self, cache_size: int = 8) -> FrameSource:
        """Frames as the lazy, random-access source the tracker contract expects."""
        if self.frames is not None:
            return ArrayFrameSource(self.frames, fps=self.fps)
        assert self.frame_dir is not None
        return DirectoryFrameSource(self.frame_dir, fps=self.fps, cache_size=cache_size)
