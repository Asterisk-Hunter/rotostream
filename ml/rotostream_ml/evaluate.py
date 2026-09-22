"""Run a registered tracker over a dataset and report DAVIS J & F.

    # toy clips, instantly, no download
    python -m rotostream_ml.evaluate --model naive --dataset synthetic

    # DAVIS 2017 val, the number you can quote
    python -m rotostream_ml.evaluate --model sam2_memory --dataset davis \
        --root D:/DAVIS --split val --limit 10 --json runs/davis_val.json

The tracker is looked up in the API's plugin registry, so this works on anything
implementing ``app.models.base.VideoObjectTracker`` - including the stub, which
will report that it is not implemented rather than silently scoring zero.

Beyond J&F, two diagnostics are reported because they are what you actually need
while building the memory stack:

* **J on frames where the object is visible vs absent.** The absent frames are the
  object-presence head's whole job: a tracker that keeps painting the last mask
  through an occlusion scores 0 there, and the split makes that visible instead of
  hiding it inside an average.
* **Presence-head accuracy / precision / recall** against the visible flag.

Prompting follows the semi-supervised protocol: one click on the first frame the
object is visible, then propagate. ``--bidirectional`` also propagates backwards
from the prompt, matching the UI's toggle.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.models import registry
from app.models.base import Direction, VideoObjectTracker

from .metrics import DatasetScore, SequenceScore, score_sequence
from .sequences import VideoSequence
from .synthetic import available as available_synthetic, build as build_synthetic

__all__ = [
    "PresenceStats",
    "SequenceEvaluation",
    "Evaluation",
    "evaluate_tracker",
    "load_sequences",
    "main",
]


@dataclass
class PresenceStats:
    """Confusion matrix for the object-presence / occlusion head."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __str__(self) -> str:
        return (
            f"accuracy {self.accuracy:.4f}  precision {self.precision:.4f}  "
            f"recall {self.recall:.4f}  (tp {self.tp} tn {self.tn} fp {self.fp} fn {self.fn})"
        )


@dataclass
class SequenceEvaluation:
    """J&F plus diagnostics for one sequence."""

    name: str
    score: SequenceScore
    presence: PresenceStats
    n_objects: int
    j_visible: float
    j_absent: float
    n_visible: int
    n_absent: int
    wall_s: float = 0.0

    @property
    def j(self) -> float:
        return self.score.j

    @property
    def f(self) -> float:
        return self.score.f

    @property
    def jf(self) -> float:
        return self.score.jf

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_frames": self.score.n_frames,
            "n_objects": self.n_objects,
            "j": self.j,
            "f": self.f,
            "jf": self.jf,
            "j_visible_frames": self.j_visible,
            "n_visible_frames": self.n_visible,
            "j_absent_frames": self.j_absent,
            "n_absent_frames": self.n_absent,
            "presence": self.presence.to_dict(),
            "wall_s": self.wall_s,
        }


@dataclass
class Evaluation:
    """Result of running one tracker over one dataset."""

    model: str
    dataset: str
    sequences: list[SequenceEvaluation] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def score(self) -> DatasetScore:
        return DatasetScore(sequences=[item.score for item in self.sequences])

    @property
    def presence(self) -> PresenceStats:
        total = PresenceStats()
        for item in self.sequences:
            total.tp += item.presence.tp
            total.tn += item.presence.tn
            total.fp += item.presence.fp
            total.fn += item.presence.fn
        return total

    def to_dict(self) -> dict:
        score = self.score
        return {
            "model": self.model,
            "dataset": self.dataset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "j": score.j,
            "f": score.f,
            "jf": score.jf,
            "n_sequences": len(self.sequences),
            "n_frames": score.n_frames,
            "elapsed_s": self.elapsed_s,
            "config": self.config,
            "presence": self.presence.to_dict(),
            "sequences": [item.to_dict() for item in self.sequences],
        }

    def report(self) -> str:
        lines = [
            "",
            f"model    {self.model}",
            f"dataset  {self.dataset}   ({len(self.sequences)} sequences, "
            f"{self.score.n_frames} frames, {self.elapsed_s:.1f}s)",
            "",
            self.score.table(),
            "",
            f"objects on screen: J {self._weighted('j_visible'):.4f} over "
            f"{sum(item.n_visible for item in self.sequences)} frames",
            f"objects absent:    J {self._weighted('j_absent'):.4f} over "
            f"{sum(item.n_absent for item in self.sequences)} frames "
            "(empty ground truth, so 1.0 means the mask was correctly empty)",
            f"presence head:     {self.presence}",
        ]
        return "\n".join(lines)

    def _weighted(self, attribute: str) -> float:
        weights = [
            (getattr(item, attribute), item.n_visible if attribute == "j_visible" else item.n_absent)
            for item in self.sequences
        ]
        total = sum(weight for _, weight in weights)
        if not total:
            return 0.0
        return sum(value * weight for value, weight in weights) / total

    def csv_rows(self) -> list[dict]:
        return [
            {
                "model": self.model,
                "dataset": self.dataset,
                "sequence": item.name,
                "n_frames": item.score.n_frames,
                "j": f"{item.j:.6f}",
                "f": f"{item.f:.6f}",
                "jf": f"{item.jf:.6f}",
                "j_visible_frames": f"{item.j_visible:.6f}",
                "j_absent_frames": f"{item.j_absent:.6f}",
                "presence_accuracy": f"{item.presence.accuracy:.6f}",
                "wall_s": f"{item.wall_s:.3f}",
            }
            for item in self.sequences
        ]


# ------------------------------------------------------------------- tracking
def _track_sequence(
    make_tracker: Callable[[], VideoObjectTracker],
    sequence: VideoSequence,
    object_index: int,
    *,
    direction: Direction,
    bidirectional: bool,
    device: str,
    checkpoint: str | None,
    prompt_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one prompt through the whole clip. Returns (masks, present, scores)."""
    prompt = sequence.prompt_for(object_index, prompt_frame)
    n_frames = sequence.n_frames
    masks = np.zeros((n_frames, sequence.height, sequence.width), dtype=bool)
    present = np.zeros(n_frames, dtype=bool)
    scores = np.zeros(n_frames, dtype=np.float32)

    tracker = make_tracker()
    tracker.load(device=device, checkpoint=checkpoint)
    tracker.set_video(sequence.frame_source())

    result = tracker.add_prompt(prompt)
    masks[prompt.frame_index] = np.asarray(result.mask)
    present[prompt.frame_index] = bool(result.object_present)
    scores[prompt.frame_index] = float(result.score)

    # `direction` is the primary sweep; --bidirectional adds the other side, which
    # is what the UI's toggle does. Neither one ever revisits the prompt frame.
    go_forward = direction is Direction.FORWARD or bidirectional
    go_backward = direction is Direction.BACKWARD or bidirectional

    if go_forward:
        for index in range(prompt.frame_index + 1, n_frames):
            result = tracker.propagate(index, Direction.FORWARD)
            masks[index] = np.asarray(result.mask)
            present[index] = bool(result.object_present)
            scores[index] = float(result.score)

    if go_backward:
        for index in range(prompt.frame_index - 1, -1, -1):
            result = tracker.propagate(index, Direction.BACKWARD)
            masks[index] = np.asarray(result.mask)
            present[index] = bool(result.object_present)
            scores[index] = float(result.score)

    return masks, present, scores


def evaluate_tracker(
    model: str | Callable[[], VideoObjectTracker],
    sequences: Sequence[VideoSequence],
    *,
    object_index: int = 0,
    direction: Direction = Direction.FORWARD,
    bidirectional: bool = False,
    device: str = "cpu",
    checkpoint: str | None = None,
    dataset: str = "synthetic",
    prompt_frame: int | None = None,
    progress: bool = True,
) -> Evaluation:
    """Evaluate ``model`` on every sequence, one object at a time.

    ``model`` is either a registered key or a factory returning a fresh tracker;
    the factory form keeps tests from having to register throwaway trackers.
    """
    make_tracker = _factory_for(model) if isinstance(model, str) else model
    label = model if isinstance(model, str) else getattr(make_tracker, "__name__", "factory")
    evaluation = Evaluation(
        model=str(label),
        dataset=dataset,
        config={
            "object_index": object_index,
            "direction": direction.value,
            "bidirectional": bidirectional,
            "device": device,
            "checkpoint": checkpoint,
            "prompt_frame": prompt_frame,
        },
    )

    started = time.perf_counter()
    for position, sequence in enumerate(sequences, start=1):
        if progress:
            print(
                f"  [{position}/{len(sequences)}] {sequence.name} "
                f"({sequence.n_frames} frames, {sequence.n_objects} object(s))",
                flush=True,
            )
        sequence_started = time.perf_counter()
        masks, present, _ = _track_sequence(
            make_tracker,
            sequence,
            object_index,
            direction=direction,
            bidirectional=bidirectional,
            device=device,
            checkpoint=checkpoint,
            prompt_frame=prompt_frame,
        )

        ground_truth = sequence.object_masks(object_index)
        score = score_sequence(
            sequence.name,
            [masks[index] for index in range(sequence.n_frames)],
            [ground_truth[index] for index in range(sequence.n_frames)],
        )

        visible = sequence.visible(object_index)
        presence = PresenceStats()
        for index in range(sequence.n_frames):
            presence.add(bool(present[index]), bool(visible[index]))

        # J split by whether the object was actually on screen: this is the line
        # that tells you if the occlusion head is doing anything at all.
        visible_js = [frame.j for frame in score.frames if visible[frame.index]]
        absent_js = [frame.j for frame in score.frames if not visible[frame.index]]

        evaluation.sequences.append(
            SequenceEvaluation(
                name=sequence.name,
                score=score,
                presence=presence,
                n_objects=sequence.n_objects,
                j_visible=float(np.mean(visible_js)) if visible_js else 0.0,
                j_absent=float(np.mean(absent_js)) if absent_js else 0.0,
                n_visible=len(visible_js),
                n_absent=len(absent_js),
                wall_s=time.perf_counter() - sequence_started,
            )
        )
    evaluation.elapsed_s = time.perf_counter() - started
    return evaluation


def _factory_for(model: str) -> Callable[[], VideoObjectTracker]:
    """A fresh tracker per sequence, so no memory bank leaks between clips."""
    cls = registry.load_class(model)

    def make() -> VideoObjectTracker:
        return cls()

    return make


# --------------------------------------------------------------------- datasets
def load_sequences(args: argparse.Namespace) -> list[VideoSequence]:
    """Build the sequence list for the CLI's ``--dataset`` choice."""
    if args.dataset == "synthetic":
        names = args.sequences or available_synthetic()
        return [build_synthetic(name) for name in names]

    if args.dataset == "davis":
        from .davis import Davis2017

        if not args.root:
            raise SystemExit("--root is required for --dataset davis")
        dataset = Davis2017(root=args.root, split=args.split, resolution=args.resolution)
        print(f"DAVIS: {len(dataset)} sequences in the {args.split} split at {dataset.root}")
        names = dataset.names[: args.limit] if args.limit else dataset.names
        return [dataset.load(name) for name in names]

    raise SystemExit(f"unknown dataset {args.dataset!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a tracker with DAVIS J & F.")
    parser.add_argument("--model", default="naive", help="registered tracker key")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "davis"])
    parser.add_argument("--sequences", nargs="*", default=None, help="synthetic scenario names")
    parser.add_argument("--root", default=None, help="DAVIS root directory")
    parser.add_argument("--split", default="val", help="DAVIS split file name")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--limit", type=int, default=None, help="first N sequences")
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument(
        "--direction", default="forward", choices=[d.value for d in Direction]
    )
    parser.add_argument(
        "--bidirectional", action="store_true", help="also propagate backwards from the prompt"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None, help="path to your model checkpoint")
    parser.add_argument("--prompt-frame", type=int, default=None)
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument("--csv", default=None, help="write per-sequence rows here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    sequences = load_sequences(args)
    if not sequences:
        print("no sequences to evaluate")
        return 1

    try:
        evaluation = evaluate_tracker(
            args.model,
            sequences,
            object_index=args.object_index,
            direction=Direction(args.direction),
            bidirectional=args.bidirectional,
            device=args.device,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            prompt_frame=args.prompt_frame,
            progress=not args.quiet,
        )
    except NotImplementedError as exc:
        # The stub raises this on purpose. Say so plainly instead of printing zeros.
        print(f"\n{args.model} is registered but not implemented yet:\n  {exc}\n")
        return 2
    except FileNotFoundError as exc:
        print(f"\n{args.model} needs a checkpoint:\n  {exc}\n")
        return 3

    print(evaluation.report())

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = evaluation.csv_rows()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
