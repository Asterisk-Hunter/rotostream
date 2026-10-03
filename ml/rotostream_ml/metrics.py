"""J & F metrics for video object segmentation (DAVIS).

These kernels follow the official DAVIS metric definitions. A benchmark also
needs matching prompts, scoring windows and object aggregation; the evaluator's
``davis`` protocol controls those choices:

* ``J`` - region similarity: intersection-over-union of the predicted and ground
  truth masks.
* ``F`` - boundary similarity: precision/recall between the two *boundary maps*,
  where a boundary pixel counts as matched if it lies within a tolerance disk of
  the other boundary. The tolerance is ``0.008 * sqrt(H^2 + W^2)`` pixels
  (``bound_th``), which is ~8px at DAVIS 480p.
* ``J&F`` - the mean of the two.

Two details that quietly change the score and are easy to get wrong:

1. ``seg2bmap`` is the Martin et al. (2003) 1-pixel XOR-of-4-neighbours boundary,
   **not** ``cv2.findContours``/Canny. The official metrics use this exact
   definition; swapping it moves F by several points.
2. Each score entry averages over frames. The DAVIS evaluator creates one entry
   per object track, then averages entries so every object has equal weight.
   Pooling frames would over-weight long sequences; averaging clips would
   under-weight objects in multi-object clips.

Sources: ``davis2017-evaluation`` (davis2017/metrics.py) and Perazzi et al.,
"A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation",
CVPR 2016. The dilation is done with scipy here rather than scikit-image; the
disk pixel set and border behavior are identical (computed using Euclidean
distance transforms), and
``test_metrics.test_matches_the_skimage_reference`` pins that down.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "seg2bmap",
    "iou",
    "boundary_f",
    "boundary_tolerance",
    "j_and_f",
    "frame_jf",
    "FrameScore",
    "SequenceScore",
    "DatasetScore",
    "score_sequence",
    "score_dataset",
    "DEFAULT_BOUND_TH",
]

#: Boundary tolerance as a fraction of the frame diagonal, per the official code.
DEFAULT_BOUND_TH = 0.008

#: One (H, W) boolean mask per object in the frame.
ObjectMasks = Sequence[np.ndarray]


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Return the 1-pixel-wide boundary map of a binary mask.

    Ported from the reference (David Martin, 2003). Boundary pixels are offset by
    1/2 pixel towards the origin from the actual segment boundary, which matters
    only in that it must be applied consistently to prediction and ground truth.
    """
    seg = np.asarray(seg)
    if seg.ndim != 2:
        raise ValueError(f"seg2bmap expects (H, W), got {seg.shape}")
    seg = seg.astype(bool)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    # `^` binds tighter than `|` in Python, so this is (seg^e) | (seg^s) | (seg^se).
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


@lru_cache(maxsize=32)
def _disk(radius: int) -> np.ndarray:
    """A filled disk structuring element, matching ``skimage.morphology.disk``.

    Cached: a 480p frame needs a radius-8 disk, and rebuilding it per frame per
    object is pure waste.
    """
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (x * x + y * y) <= radius * radius
    disk.flags.writeable = False
    return disk


def boundary_tolerance(shape: tuple[int, int], bound_th: float = DEFAULT_BOUND_TH) -> int:
    """Dilation radius in pixels for boundary matching.

    ``bound_th >= 1`` is treated as an absolute pixel radius; below that it is a
    fraction of the frame diagonal (the official behaviour).
    """
    if bound_th >= 1:
        return int(bound_th)
    return int(np.ceil(bound_th * float(np.linalg.norm(shape[:2]))))


def iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Region similarity (J). Empty-vs-empty scores 1.0, the official convention."""
    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {ground_truth.shape}")

    union = int(np.logical_or(prediction, ground_truth).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, ground_truth).sum() / union)


def boundary_f(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    bound_th: float = DEFAULT_BOUND_TH,
    tolerance: int | None = None,
) -> float:
    """Boundary similarity (F), with the official precision/recall conventions.

    ``tolerance`` overrides the derived dilation radius; pass the value from
    :func:`boundary_tolerance` to avoid recomputing it per frame.
    """
    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {ground_truth.shape}")

    radius = boundary_tolerance(prediction.shape, bound_th) if tolerance is None else int(tolerance)
    fg_boundary = seg2bmap(prediction)
    gt_boundary = seg2bmap(ground_truth)

    n_fg = int(fg_boundary.sum())
    n_gt = int(gt_boundary.sum())

    if n_fg == 0 and n_gt > 0:
        return 0.0
    elif n_fg > 0 and n_gt == 0:
        return 0.0
    elif n_fg == 0 and n_gt == 0:
        return 1.0
    else:
        # A disk dilation is exactly the set of pixels at Euclidean distance
        # <= radius from the boundary. EDT avoids visiting every disk pixel for
        # every background pixel (~8x faster at DAVIS 480p), preserving the
        # official metric including image-edge behavior.
        fg_dilated = ndimage.distance_transform_edt(~fg_boundary) <= radius
        gt_dilated = ndimage.distance_transform_edt(~gt_boundary) <= radius
        precision = float(np.sum(fg_boundary & gt_dilated)) / n_fg
        recall = float(np.sum(gt_boundary & fg_dilated)) / n_gt

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def j_and_f(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    bound_th: float = DEFAULT_BOUND_TH,
    tolerance: int | None = None,
) -> tuple[float, float]:
    """Region similarity and boundary similarity for one mask pair."""
    return (
        iou(prediction, ground_truth),
        boundary_f(prediction, ground_truth, bound_th, tolerance),
    )


def frame_jf(
    predictions: ObjectMasks, ground_truths: ObjectMasks, bound_th: float = DEFAULT_BOUND_TH
) -> tuple[float, float]:
    """Mean J and F over the objects in a single frame.

    A frame with no objects at all throws: that is almost always a bug in the
    caller (DAVIS indexes objects from 1, so a misread annotation looks like an
    empty frame), and silently scoring it 1.0 would inflate the result.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"object count mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )
    if not len(ground_truths):
        raise ValueError("frame has no objects")

    # Every object in a frame shares the frame's shape, so the dilation radius
    # and its structuring element are computed once here rather than per object.
    tolerance = boundary_tolerance(np.asarray(ground_truths[0]).shape, bound_th)
    js, fs = [], []
    for prediction, ground_truth in zip(predictions, ground_truths):
        j, f = j_and_f(prediction, ground_truth, bound_th, tolerance)
        js.append(j)
        fs.append(f)
    return float(np.mean(js)), float(np.mean(fs))


@dataclass(frozen=True)
class FrameScore:
    """J/F for one frame, averaged over that frame's objects."""

    index: int
    j: float
    f: float
    n_objects: int

    @property
    def jf(self) -> float:
        return 0.5 * (self.j + self.f)

    def to_dict(self) -> dict:
        return {"index": self.index, "j": self.j, "f": self.f, "jf": self.jf, "n_objects": self.n_objects}


@dataclass
class SequenceScore:
    """Per-sequence J/F, aggregated as the mean over frames."""

    name: str
    frames: list[FrameScore] = field(default_factory=list)

    @property
    def j(self) -> float:
        return float(np.mean([frame.j for frame in self.frames])) if self.frames else 0.0

    @property
    def f(self) -> float:
        return float(np.mean([frame.f for frame in self.frames])) if self.frames else 0.0

    @property
    def jf(self) -> float:
        return 0.5 * (self.j + self.f)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_frames": self.n_frames,
            "j": self.j,
            "f": self.f,
            "jf": self.jf,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass
class DatasetScore:
    """J/F averaged over entries (one object track per entry for DAVIS evaluation)."""

    sequences: list[SequenceScore] = field(default_factory=list)

    @property
    def j(self) -> float:
        return float(np.mean([s.j for s in self.sequences])) if self.sequences else 0.0

    @property
    def f(self) -> float:
        return float(np.mean([s.f for s in self.sequences])) if self.sequences else 0.0

    @property
    def jf(self) -> float:
        return 0.5 * (self.j + self.f)

    @property
    def n_frames(self) -> int:
        return sum(s.n_frames for s in self.sequences)

    def to_dict(self) -> dict:
        return {
            "n_sequences": len(self.sequences),
            "n_frames": self.n_frames,
            "j": self.j,
            "f": self.f,
            "jf": self.jf,
            "sequences": [s.to_dict() for s in self.sequences],
        }

    def table(self) -> str:
        """A fixed-width per-sequence summary, suitable for the console."""
        if not self.sequences:
            return "(no sequences)"
        width = max(len(s.name) for s in self.sequences)
        lines = [f"{'sequence'.ljust(width)}  frames      J      F    J&F"]
        for score in sorted(self.sequences, key=lambda s: -(0.5 * (s.j + s.f))):
            lines.append(
                f"{score.name.ljust(width)}  {score.n_frames:6d}  {score.j:6.4f} "
                f"{score.f:6.4f} {score.jf:6.4f}"
            )
        lines.append("-" * (width + 30))
        lines.append(
            f"{'MEAN'.ljust(width)}  {self.n_frames:6d}  {self.j:6.4f} "
            f"{self.f:6.4f} {self.jf:6.4f}"
        )
        return "\n".join(lines)


def _as_object_list(masks: np.ndarray | ObjectMasks, n_objects: int) -> list[np.ndarray]:
    """Accept either an (N, H, W) stack or a list of (H, W) masks."""
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        return [masks[i] for i in range(masks.shape[0])]
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        return [masks]
    listed = list(masks)
    if len(listed) == 1 and np.asarray(listed[0]).ndim == 3:
        return _as_object_list(np.asarray(listed[0]), n_objects)
    return [np.asarray(mask) for mask in listed]


def score_sequence(
    name: str,
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
    bound_th: float = DEFAULT_BOUND_TH,
) -> SequenceScore:
    """Score one sequence frame by frame.

    ``predictions`` and ``ground_truth`` are per-frame; each frame is either a
    single (H, W) mask, an (N, H, W) stack, or a list of (H, W) masks. The
    ground truth defines the object count, and predictions must match it - a
    mismatch is an error rather than something to paper over, because it usually
    means objects were dropped or the annotation was read wrong.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"frame count mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground truths"
        )

    score = SequenceScore(name=name)
    for index, (predicted, truth) in enumerate(zip(predictions, ground_truth)):
        gt_objects = _as_object_list(truth, 0)
        pred_objects = _as_object_list(predicted, 0)
        if len(pred_objects) != len(gt_objects):
            raise ValueError(
                f"frame {index} of '{name}': {len(pred_objects)} predicted objects vs "
                f"{len(gt_objects)} ground truth"
            )
        j, f = frame_jf(pred_objects, gt_objects, bound_th)
        score.frames.append(FrameScore(index=index, j=j, f=f, n_objects=len(gt_objects)))
    return score


def score_dataset(
    sequences: Sequence[tuple[str, Sequence[np.ndarray], Sequence[np.ndarray]]],
    bound_th: float = DEFAULT_BOUND_TH,
) -> DatasetScore:
    """Score ``[(name, predictions, ground_truth), ...]`` into a DatasetScore."""
    return DatasetScore(
        sequences=[
            score_sequence(name, predictions, ground_truth, bound_th)
            for name, predictions, ground_truth in sequences
        ]
    )
