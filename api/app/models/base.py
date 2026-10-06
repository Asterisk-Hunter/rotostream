"""The RotoStream tracker contract.

**This is the only file you need to read to plug in a model.** Everything else in
the repository (API, UI, export, evaluation, training harness) is written against
the interfaces defined here.

The application drives a tracker like this::

    tracker.load(device="auto", checkpoint=None)   # once
    tracker.set_video(frame_source)                # once per video
    tracker.add_prompt(PromptSet(0, points=(PointPrompt(x, y, True),)))
    result = tracker.propagate(1, Direction.FORWARD)
    result = tracker.propagate(2, Direction.FORWARD)
    ...

A minimal, complete implementation
----------------------------------

    from app.models.base import (
        Direction, FrameResult, FrameSource, PromptSet, TrackerInfo, VideoObjectTracker,
    )

    class AlwaysCentre(VideoObjectTracker):
        key = "always_centre"

        @classmethod
        def info(cls) -> TrackerInfo:
            return TrackerInfo(name=cls.key, description="trivial demo tracker")

        def load(self, *, device="auto", checkpoint=None):
            self._frames = None

        def set_video(self, frames: FrameSource) -> None:
            self._frames = frames

        def add_prompt(self, prompts: PromptSet) -> FrameResult:
            return self._mask(prompts.frame_index)

        def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
            return self._mask(frame_index)

        def _mask(self, index: int) -> FrameResult:
            h, w = self._frames.shape
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
            return FrameResult(mask=mask)

Rules the application relies on
-------------------------------
1. ``load`` is called exactly once, before ``set_video``.
2. ``set_video`` is called once per video. It may precompute per-frame image
   embeddings, but must not require the entire video in RAM — ``frames`` is lazy
   and random-access.
3. ``add_prompt`` must store the prompted frame into the memory bank and return
   the mask **for that same frame**.
4. ``propagate(frame_index, direction)`` must condition on memory from
   **already-visited frames only**: for ``FORWARD``, memory from indices
   ``< frame_index``; for ``BACKWARD``, memory from indices ``> frame_index``.
   Reading from the unvisited side is future leakage — it inflates benchmark
   scores and is caught by ``api/tests/test_contract.py``.
5. Masks are ``(H, W)`` ``bool`` at the ``FrameSource`` resolution. When
   ``object_present`` is ``False`` the mask must be entirely ``False``.
6. ``propagate`` is called at most once per frame per direction, in order, but
   must tolerate being called again for the same frame (the pipeline can retry).
7. ``reset()`` clears the memory bank and all prompts so the session can restart.

Run ``python api/scripts/check_model.py <key>`` to verify a plugin against these
rules, including the future-leakage check.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .frames import FrameSource

__all__ = [
    "Direction",
    "PointPrompt",
    "BoxPrompt",
    "PromptSet",
    "FrameResult",
    "TrackerInfo",
    "TrainingSequence",
    "VideoObjectTracker",
    "TrainableTracker",
    "ContractError",
    "resolve_device",
    "check_frame_result",
]


class ContractError(RuntimeError):
    """Raised when a tracker violates the contract in a way the app cannot recover from."""


class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class PointPrompt:
    """A click. ``positive=False`` marks background the object must not cover."""

    x: float
    y: float
    positive: bool = True


@dataclass(frozen=True)
class BoxPrompt:
    """An axis-aligned box hint, in pixel coordinates of the prompted frame."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)


@dataclass(frozen=True)
class PromptSet:
    """All prompts that apply to a single frame."""

    frame_index: int
    points: tuple[PointPrompt, ...] = ()
    box: BoxPrompt | None = None
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not self.points and self.box is None and self.mask is None:
            raise ValueError("a PromptSet needs at least one point, a box or a mask")
        if self.mask is not None:
            if self.points or self.box is not None:
                raise ValueError("a mask prompt cannot be combined with points or a box")
            mask = np.asarray(self.mask)
            if mask.ndim != 2 or not all(mask.shape) or mask.dtype != np.bool_:
                raise ValueError("mask prompt must be a nonempty (H, W) boolean array")
            mask = np.array(mask, dtype=bool, copy=True, order="C")
            mask.flags.writeable = False
            object.__setattr__(self, "mask", mask)

    @property
    def positive_points(self) -> tuple[PointPrompt, ...]:
        return tuple(p for p in self.points if p.positive)

    @property
    def negative_points(self) -> tuple[PointPrompt, ...]:
        return tuple(p for p in self.points if not p.positive)


@dataclass
class FrameResult:
    """What a tracker returns for a single frame."""

    #: (H, W) boolean array, True where the object is.
    mask: np.ndarray
    #: Predicted mask quality in [0, 1]. Surfaced in the UI and used to flag drift.
    score: float = 1.0
    #: Output of the object-presence / occlusion head. False means "not in frame".
    object_present: bool = True
    #: Free-form diagnostics (e.g. logits, memory indices). Never required.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mask = np.ascontiguousarray(self.mask, dtype=bool)
        self.score = float(self.score)


@dataclass(frozen=True)
class TrackerInfo:
    """Static description of a plugin, shown in the UI and ``GET /api/models``."""

    name: str
    description: str = ""
    #: True if the implementation keeps a memory bank across frames.
    uses_memory: bool = False
    #: True if it exposes trainable parameters via ``TrainableTracker``.
    trainable: bool = False
    #: False while a stub is still unimplemented; such plugins are hidden by the UI.
    implemented: bool = True
    #: How to obtain the checkpoint this plugin expects, shown as a hint in the UI.
    checkpoint_hint: str = ""
    #: Populated by the registry when a plugin could not be imported at all.
    error: str = ""
    #: True when a prompt containing only background clicks is a valid instruction
    #: ("there is no object on this frame") rather than a contract violation. The
    #: studio uses it to warn before a run instead of failing inside the worker.
    accepts_background_only_prompts: bool = False


def resolve_device(requested: str = "auto") -> str:
    """Map ``"auto"`` to ``"cuda"`` when torch reports a usable GPU, else ``"cpu"``.

    Deliberately does not import torch unless needed: the reference tracker, the
    API and the test-suite all run with no torch installed.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def check_frame_result(result: FrameResult, expected_shape: tuple[int, int]) -> None:
    """Verify a ``FrameResult`` matches the frame it claims to describe."""
    if not isinstance(result, FrameResult):
        raise ContractError(
            f"tracker returned {type(result).__name__}, expected FrameResult"
        )
    mask = np.asarray(result.mask)
    if mask.shape != tuple(expected_shape):
        raise ContractError(
            f"mask shape {mask.shape} != frame shape {tuple(expected_shape)}; "
            "masks must be at FrameSource resolution, not model resolution"
        )
    if mask.dtype != np.bool_:
        raise ContractError(f"mask dtype {mask.dtype} != bool")
    if not 0.0 <= float(result.score) <= 1.0:
        raise ContractError(f"score {result.score} outside [0, 1]")
    if not result.object_present and mask.any():
        raise ContractError(
            "object_present=False but the mask is non-empty; the occlusion head must "
            "return an all-False mask when the object leaves the frame"
        )


@dataclass
class TrainingSequence:
    """One training sample handed to ``TrainableTracker.training_step``."""

    #: (T, H, W, 3) uint8 RGB.
    frames: np.ndarray
    #: (T, H, W) bool ground truth.
    gt_masks: np.ndarray
    #: Prompts to condition on, usually just the first frame.
    prompts: tuple[PromptSet, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames.ndim != 4:
            raise ValueError(f"frames must be (T, H, W, 3), got {self.frames.shape}")
        if self.gt_masks.ndim != 3:
            raise ValueError(f"gt_masks must be (T, H, W), got {self.gt_masks.shape}")
        if self.frames.shape[0] != self.gt_masks.shape[0]:
            raise ValueError("frames and gt_masks must have the same length")


class VideoObjectTracker(abc.ABC):
    """Contract for a video object tracker. Implement these five methods.

    Subclasses must also set the class attribute ``key`` to the registry name.
    """

    #: Registry key, e.g. ``"sam2_memory"``. Must be unique.
    key: str = "abstract"

    @classmethod
    def info(cls) -> TrackerInfo:
        """Describe this plugin. Override to surface memory/trainable flags."""
        return TrackerInfo(name=cls.key, description=(cls.__doc__ or "").strip().split("\n")[0])

    # ------------------------------------------------------------------ lifecycle
    @abc.abstractmethod
    def load(self, *, device: str = "auto", checkpoint: str | Path | None = None) -> None:
        """Bring up weights. Use ``resolve_device(device)`` for ``"auto"``.

        Raise ``FileNotFoundError`` with an actionable message if a checkpoint is
        required and missing, or ``NotImplementedError`` while unimplemented.
        """

    @abc.abstractmethod
    def set_video(self, frames: FrameSource) -> None:
        """Attach a video for the session. ``frames`` is lazy and random-access."""

    # ------------------------------------------------------------------- tracking
    @abc.abstractmethod
    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        """Consume prompts on ``prompts.frame_index``.

        Must (a) encode that frame, (b) push it into the memory bank, and
        (c) return the mask for that frame.
        """

    @abc.abstractmethod
    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        """Predict ``frame_index`` and push it into the memory bank.

        Must only attend to memory from frames already visited in ``direction``.
        """

    # ------------------------------------------------------------------- optional
    def reset(self) -> None:
        """Clear the memory bank and prompts. Called between sessions."""

    def memory_state(self) -> dict[str, Any]:
        """Diagnostics for the UI memory inspector, e.g.::

            {"bank_size": 5, "frames": [0, 1, 2, 3, 4], "prompted": [0], "detail": "..."}
        """
        return {}

    def warmup(self) -> None:
        """Optional: run a dummy forward pass so the first real frame is not slow."""


class TrainableTracker(VideoObjectTracker):
    """Optional extra contract so ``ml/train.py`` can train your memory stack.

    Implement this in addition to ``VideoObjectTracker`` if you want the provided
    training loop to drive your model. Only the *model internals* are yours: the
    harness owns the data pipeline, schedule, augmentations, checkpointing and
    logging.
    """

    #: Path or HF id the harness should suggest when no checkpoint is given.
    checkpoint_hint: str = ""

    @abc.abstractmethod
    def trainable_parameters(self) -> Iterable[Any]:
        """Return the parameters to optimise (typically everything except the frozen
        image encoder)."""

    @abc.abstractmethod
    def training_step(self, sequence: TrainingSequence) -> dict[str, Any]:
        """Run a differentiable forward pass and return at least ``{"loss": scalar}``.

        Any other tensor values in the returned dict are logged as scalar metrics.
        """
