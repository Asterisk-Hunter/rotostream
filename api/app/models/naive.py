"""Reference tracker: per-frame colour-similarity segmentation. **Not** the model.

This exists for two reasons:

1. It makes the entire application usable before the real tracker is written —
   upload, prompt, scrub, export all work end to end.
2. It is the *baseline your memory-attention tracker must beat*. It has **no
   memory bank** and no occlusion head, so it demonstrates exactly the failure
   modes described in the project README: the seed colour is frozen at the
   prompted frame so it drifts under lighting change, and after an occlusion it
   re-finds "whatever looks similar" rather than the tracked identity.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import (
    ContractError,
    Direction,
    FrameResult,
    FrameSource,
    PointPrompt,
    PromptSet,
    TrackerInfo,
    VideoObjectTracker,
    resolve_device,
)

SEARCH_RADIUS = 7


class NaiveColorTracker(VideoObjectTracker):
    """Frozen-seed colour flood segmentation, recomputed independently per frame."""

    key = "naive"

    def __init__(self, tolerance: float = 30.0, min_area_px: int = 48):
        self.tolerance = float(tolerance)
        self.min_area_px = int(min_area_px)
        self._frames: FrameSource | None = None
        self._seed_lab: np.ndarray | None = None
        self._positives: list[PointPrompt] = []
        self._negatives: list[PointPrompt] = []
        self._prompt_frame: int | None = None
        self._ref_area: float = 0.0
        self._device = "cpu"
        self._last_score = 1.0

    # ------------------------------------------------------------------ metadata
    @classmethod
    def info(cls) -> TrackerInfo:
        return TrackerInfo(
            name=cls.key,
            description=(
                "Reference baseline: colour similarity, re-segmented per frame. "
                "No memory bank, no occlusion head."
            ),
            uses_memory=False,
            trainable=False,
            implemented=True,
        )

    # ----------------------------------------------------------------- lifecycle
    def load(self, *, device: str = "auto", checkpoint: str | Path | None = None) -> None:
        # Nothing to load: this runs on OpenCV/NumPy. Kept for contract parity.
        self._device = resolve_device(device)

    def set_video(self, frames: FrameSource) -> None:
        self._frames = frames
        self.reset()

    def reset(self) -> None:
        self._seed_lab = None
        self._positives = []
        self._negatives = []
        self._prompt_frame = None
        self._ref_area = 0.0
        self._last_score = 1.0

    # ------------------------------------------------------------------ tracking
    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        self._require_video()
        assert self._frames is not None

        if prompts.mask is not None:
            mask = prompts.mask
            if mask.shape != self._frames.shape or not mask.any():
                raise ContractError("a colour mask prompt must match the frame and contain foreground")
            # Fit the same frozen colour baseline from the supplied object region.
            # The prompted frame stays exact; subsequent frames use the ordinary
            # colour segmentation path, so GT masks do not leak into propagation.
            lab = self._to_lab(self._frames[prompts.frame_index])
            self._seed_lab = np.median(lab[mask], axis=0)
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(distance.argmax()), mask.shape)
            self._positives = [PointPrompt(float(x), float(y), True)]
            self._negatives = []
            self._prompt_frame = prompts.frame_index
            self._ref_area = float(mask.sum())
            self._last_score = 1.0
            return FrameResult(mask=np.array(mask), score=1.0, object_present=True)

        positives = list(prompts.positive_points)
        if not positives and prompts.box is not None:
            box = prompts.box
            positives = [
                PointPrompt((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0, True)
            ]
        if not positives:
            raise ContractError(
                "Mark the object on this frame before excluding background. "
                "The colour tracker needs a foreground point or a box on each prompted frame."
            )

        self._positives = positives
        self._negatives = list(prompts.negative_points)
        self._prompt_frame = prompts.frame_index

        frame = self._frames[prompts.frame_index]
        lab = self._to_lab(frame)
        self._seed_lab = self._sample_seed(lab, positives)

        mask = self._segment(lab)
        self._ref_area = float(mask.sum())
        self._last_score = 1.0
        return FrameResult(mask=mask, score=1.0, object_present=bool(mask.any()))

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        self._require_video()
        assert self._frames is not None
        if self._seed_lab is None:
            raise ContractError("add_prompt() must be called before propagate()")

        # No memory bank is consulted - that is the point of this baseline.
        lab = self._to_lab(self._frames[frame_index])
        mask = self._segment(lab)
        area = float(mask.sum())
        score = self._area_score(area)
        self._last_score = score
        return FrameResult(mask=mask, score=score, object_present=bool(mask.any()))

    def memory_state(self) -> dict:
        return {
            "bank_size": 0,
            "frames": [] if self._prompt_frame is None else [self._prompt_frame],
            "prompted": [] if self._prompt_frame is None else [self._prompt_frame],
            "detail": (
                "No memory bank. Seed colour frozen at the prompted frame, "
                "so this deliberately drifts."
            ),
        }

    # ------------------------------------------------------------------- internals
    @staticmethod
    def _to_lab(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)

    @staticmethod
    def _sample_seed(lab: np.ndarray, positives: list[PointPrompt]) -> np.ndarray:
        """Median Lab colour in a small window around the mean positive click.

        Sampling a window rather than a single pixel keeps the seed from locking
        onto noise or an antialiased edge pixel.
        """
        h, w = lab.shape[:2]
        cx = int(round(float(np.mean([p.x for p in positives]))))
        cy = int(round(float(np.mean([p.y for p in positives]))))
        x0, x1 = max(0, cx - 3), min(w, cx + 4)
        y0, y1 = max(0, cy - 3), min(h, cy + 4)
        if x0 >= x1 or y0 >= y1:
            raise ContractError(f"prompt ({cx}, {cy}) lies outside the frame ({w}x{h})")
        window = lab[y0:y1, x0:x1].reshape(-1, 3)
        return np.median(window, axis=0)

    def _segment(self, lab: np.ndarray) -> np.ndarray:
        assert self._seed_lab is not None
        distance = np.linalg.norm(lab - self._seed_lab[None, None, :], axis=2)
        binary = (distance <= self.tolerance).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        _, labels = cv2.connectedComponents(binary, connectivity=8)
        keep = np.zeros(labels.shape, dtype=bool)

        for point in self._positives:
            label = self._label_at(labels, point)
            if label > 0:
                keep |= labels == label

        for point in self._negatives:
            label = self._label_at(labels, point)
            if label > 0:
                keep &= labels != label

        if keep.sum() < self.min_area_px and not self._negatives:
            keep = binary.astype(bool)
        return keep

    @staticmethod
    def _label_at(labels: np.ndarray, point: PointPrompt) -> int:
        """Label under a click, searching outward when the click lands off-mask."""
        h, w = labels.shape
        cx, cy = int(round(point.x)), int(round(point.y))
        for radius in range(SEARCH_RADIUS + 1):
            x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            window = labels[y0:y1, x0:x1]
            found = window[window > 0]
            if found.size:
                values, counts = np.unique(found, return_counts=True)
                return int(values[np.argmax(counts)])
        return 0

    def _area_score(self, area: float) -> float:
        """Heuristic confidence: how far the mask area has drifted from the prompt.

        ``1.0`` when the area matches the prompted frame, decaying toward ``0`` as
        it grows/shrinks by orders of magnitude. This is a stand-in for the
        predicted-IoU head a real tracker would expose.
        """
        if self._ref_area <= 0:
            return 1.0 if area > 0 else 0.0
        ratio = max(area, 1.0) / self._ref_area
        return float(np.clip(1.0 - abs(np.log(ratio)), 0.0, 1.0))

    def _require_video(self) -> None:
        if self._frames is None:
            raise ContractError("set_video() must be called before tracking")
