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

DAVIS defaults to first-frame ground-truth masks, all objects and the official
interior-frame scoring window. Synthetic runs default to a reproducible interior
click. ``--bidirectional`` runs two causal sweeps; it is not future conditioning.
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

from .metrics import DatasetScore, FrameScore, SequenceScore, score_sequence
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
    object_index: int = 0
    prompt_frame: int = 0
    scored_frame_indices: list[int] = field(default_factory=list)

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
            "object_index": self.object_index,
            "prompt_frame": self.prompt_frame,
            "scored_frame_indices": self.scored_frame_indices,
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
            "n_sequences": self.config.get("n_sequences", len(self.sequences)),
            "n_object_tracks": len(self.sequences),
            "score_unit": "fraction",
            "n_frames": score.n_frames,
            "n_scored_object_frames": score.n_frames,
            "elapsed_s": self.elapsed_s,
            "config": self.config,
            "presence": self.presence.to_dict(),
            "sequences": [item.to_dict() for item in self.sequences],
        }

    def report(self) -> str:
        lines = [
            "",
            f"model    {self.model}",
            f"dataset  {self.dataset}   ({self.config.get('n_sequences', len(self.sequences))} sequences, "
            f"{len(self.sequences)} object tracks, {self.score.n_frames} scored object-frames, {self.elapsed_s:.1f}s)",
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
                "object_index": item.object_index,
                "prompt_mode": self.config.get("prompt_mode", "point"),
                "protocol": self.config.get("protocol", "diagnostic"),
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
    prompt_mode: str = "point",
    tracker: VideoObjectTracker | None = None,
    timings: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one prompt through the whole clip. Returns (masks, present, scores)."""
    prompt = sequence.prompt_for(object_index, prompt_frame, mode=prompt_mode)
    n_frames = sequence.n_frames
    masks = np.zeros((n_frames, sequence.height, sequence.width), dtype=bool)
    present = np.zeros(n_frames, dtype=bool)
    scores = np.zeros(n_frames, dtype=np.float32)

    if tracker is None:
        tracker = make_tracker()
        tracker.load(device=device, checkpoint=checkpoint)
    tracker.set_video(sequence.frame_source())

    operation_started = time.perf_counter()
    result = tracker.add_prompt(prompt)
    if timings is not None:
        timings["prompt_s"] += time.perf_counter() - operation_started
        timings["prompt_frames"] += 1
    masks[prompt.frame_index] = np.asarray(result.mask)
    present[prompt.frame_index] = bool(result.object_present)
    scores[prompt.frame_index] = float(result.score)

    # `direction` is the primary sweep; --bidirectional adds the other side, which
    # is what the UI's toggle does. Neither one ever revisits the prompt frame.
    go_forward = direction is Direction.FORWARD or bidirectional
    go_backward = direction is Direction.BACKWARD or bidirectional

    if go_forward:
        for index in range(prompt.frame_index + 1, n_frames):
            operation_started = time.perf_counter()
            result = tracker.propagate(index, Direction.FORWARD)
            if timings is not None:
                timings["propagation_s"] += time.perf_counter() - operation_started
                timings["propagated_frames"] += 1
            masks[index] = np.asarray(result.mask)
            present[index] = bool(result.object_present)
            scores[index] = float(result.score)

    if go_backward:
        for index in range(prompt.frame_index - 1, -1, -1):
            operation_started = time.perf_counter()
            result = tracker.propagate(index, Direction.BACKWARD)
            if timings is not None:
                timings["propagation_s"] += time.perf_counter() - operation_started
                timings["propagated_frames"] += 1
            masks[index] = np.asarray(result.mask)
            present[index] = bool(result.object_present)
            scores[index] = float(result.score)

    return masks, present, scores


def evaluate_tracker(
    model: str | Callable[[], VideoObjectTracker],
    sequences: Sequence[VideoSequence],
    *,
    object_index: int | None = 0,
    direction: Direction = Direction.FORWARD,
    bidirectional: bool = False,
    device: str = "cpu",
    checkpoint: str | None = None,
    dataset: str = "synthetic",
    prompt_frame: int | None = None,
    prompt_mode: str = "point",
    protocol: str = "diagnostic",
    model_options: dict | None = None,
    reuse_tracker: bool = False,
    progress: bool = True,
) -> Evaluation:
    """Evaluate ``model`` on every sequence, one object at a time.

    ``model`` is either a registered key or a factory returning a fresh tracker;
    the factory form keeps tests from having to register throwaway trackers.
    """
    if protocol not in {"diagnostic", "davis"}:
        raise ValueError("protocol must be diagnostic or davis")
    if prompt_mode not in {"point", "mask"}:
        raise ValueError("prompt_mode must be point or mask")
    if protocol == "davis":
        if prompt_mode != "mask" or direction is not Direction.FORWARD or bidirectional or prompt_frame not in {None, 0}:
            raise ValueError("DAVIS protocol requires a frame-0 mask prompt and one forward sweep")
        if object_index is not None:
            raise ValueError("DAVIS protocol must evaluate all objects (object_index=None)")
        prompt_frame = 0
    make_tracker = _factory_for(model, model_options) if isinstance(model, str) else model
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
            "prompt_mode": prompt_mode,
            "protocol": protocol,
            "aggregation": "mean_over_object_tracks",
            "multi_object_policy": "independent_per_object_sessions_without_joint_pixel_logit_arbitration",
            "scoring_window": "frames_1_through_T_minus_2" if protocol == "davis" else "visited_frames",
            "model_options": model_options or {},
            "n_sequences": len(sequences),
            "sequence_names": [sequence.name for sequence in sequences] if isinstance(sequences, list) else [],
            "timings": {"prompt_s": 0.0, "prompt_frames": 0, "propagation_s": 0.0, "propagated_frames": 0},
        },
    )

    started = time.perf_counter()
    shared_tracker = None
    if reuse_tracker:
        shared_tracker = make_tracker()
        shared_tracker.load(device=device, checkpoint=checkpoint)
        evaluation.config["weights_id"] = getattr(shared_tracker, "backbone_name", None)
        evaluation.config["weights_revision"] = getattr(getattr(shared_tracker, "config", None), "_commit_hash", None)
        evaluation.config["working_resolution"] = getattr(getattr(shared_tracker, "stack", None), "image_size", None)
    for position, sequence in enumerate(sequences, start=1):
        if progress:
            print(
                f"  [{position}/{len(sequences)}] {sequence.name} "
                f"({sequence.n_frames} frames, {sequence.n_objects} object(s))",
                flush=True,
            )
        for selected_object in (range(sequence.n_objects) if object_index is None else [object_index]):
            _evaluate_object(evaluation, make_tracker, sequence, selected_object,
                             direction=direction, bidirectional=bidirectional, device=device,
                             checkpoint=checkpoint, prompt_frame=prompt_frame, prompt_mode=prompt_mode,
                             protocol=protocol, tracker=shared_tracker)
    evaluation.elapsed_s = time.perf_counter() - started
    timings = evaluation.config["timings"]
    timings["propagation_fps"] = timings["propagated_frames"] / timings["propagation_s"] if timings["propagation_s"] else 0.0
    timings["fps_excludes"] = ["weight_loading", "prompt_frames", "dataset_loading", "metric_scoring"]
    return evaluation


def _evaluate_object(evaluation, make_tracker, sequence, object_index, *, direction,
                     bidirectional, device, checkpoint, prompt_frame, prompt_mode, protocol,
                     tracker=None):
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
        prompt_mode=prompt_mode,
        tracker=tracker,
        timings=evaluation.config["timings"],
    )

    ground_truth = sequence.object_masks(object_index)
    actual_prompt = sequence.first_visible_frame(object_index) if prompt_frame is None else prompt_frame
    if protocol == "davis":
        indices = list(range(1, sequence.n_frames - 1))
    elif bidirectional:
        indices = list(range(sequence.n_frames))
    elif direction is Direction.FORWARD:
        indices = list(range(actual_prompt, sequence.n_frames))
    else:
        indices = list(range(actual_prompt + 1))
    if not indices:
        raise ValueError(f"{sequence.name}: no frames in the scoring window")
    name = sequence.name if evaluation.config["object_index"] is not None else f"{sequence.name}/object_{object_index + 1}"
    score = score_sequence(name, [masks[index] for index in indices],
                           [ground_truth[index] for index in indices])
    score.frames = [FrameScore(index=index, j=frame.j, f=frame.f, n_objects=frame.n_objects)
                    for index, frame in zip(indices, score.frames)]

    visible = sequence.visible(object_index)
    presence = PresenceStats()
    for index in indices:
        presence.add(bool(present[index]), bool(visible[index]))

    # J split by whether the object was actually on screen: this is the line
    # that tells you if the occlusion head is doing anything at all.
    visible_js = [frame.j for frame in score.frames if visible[frame.index]]
    absent_js = [frame.j for frame in score.frames if not visible[frame.index]]

    evaluation.sequences.append(
        SequenceEvaluation(
            name=name,
            score=score,
            presence=presence,
            n_objects=sequence.n_objects,
            j_visible=float(np.mean(visible_js)) if visible_js else 0.0,
            j_absent=float(np.mean(absent_js)) if absent_js else 0.0,
            n_visible=len(visible_js),
            n_absent=len(absent_js),
            wall_s=time.perf_counter() - sequence_started,
            object_index=object_index,
            prompt_frame=actual_prompt,
            scored_frame_indices=indices,
        )
    )


def _factory_for(model: str, options: dict | None = None) -> Callable[[], VideoObjectTracker]:
    """A fresh tracker per sequence, so no memory bank leaks between clips."""
    cls = registry.load_class(model)

    def make() -> VideoObjectTracker:
        return cls(**(options or {}))

    return make


# --------------------------------------------------------------------- datasets
class _DavisSequences(Sequence[VideoSequence]):
    """Materialise one clip at a time, keeping a full validation run within RAM."""

    def __init__(self, dataset, names):
        self.dataset, self.names = dataset, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.dataset.load(name) for name in self.names[index]]
        return self.dataset.load(self.names[index])


def load_sequences(args: argparse.Namespace) -> Sequence[VideoSequence]:
    """Build the sequence list for the CLI's ``--dataset`` choice."""
    if args.dataset == "synthetic":
        names = args.sequences or available_synthetic()
        return [build_synthetic(name) for name in names]

    if args.dataset == "davis":
        from .davis import Davis2017

        if not args.root:
            raise SystemExit("--root is required for --dataset davis")
        dataset = Davis2017(root=args.root, split=args.split, resolution=args.resolution,
                            strict=args.protocol == "davis")
        print(f"DAVIS: {len(dataset)} sequences in the {args.split} split at {dataset.root}")
        names = dataset.names[: args.limit] if args.limit else dataset.names
        return _DavisSequences(dataset, names)

    raise SystemExit(f"unknown dataset {args.dataset!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a tracker with DAVIS J & F.")
    parser.add_argument("--model", default="naive", help="registered tracker key")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "davis"])
    parser.add_argument("--sequences", nargs="*", choices=available_synthetic(), default=None, help="synthetic scenario names")
    parser.add_argument("--root", default=None, help="DAVIS root directory")
    parser.add_argument("--split", default="val", help="DAVIS split file name")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--limit", type=int, default=None, help="first N sequences")
    parser.add_argument("--object-index", type=int, default=None, help="diagnostic mode: score one object; otherwise all")
    parser.add_argument("--prompt-mode", choices=["point", "mask"], default=None)
    parser.add_argument("--protocol", choices=["diagnostic", "davis"], default=None,
                        help="DAVIS: all objects, first-frame masks, exclude first/last frames")
    parser.add_argument("--backbone", default=None, help="SAM 2 Hugging Face model id")
    parser.add_argument("--memory-bank-size", type=int, choices=range(1, 9), default=None)
    parser.add_argument("--no-presence-head", action="store_true")
    parser.add_argument("--no-object-pointers", action="store_true")
    parser.add_argument("--no-temporal-position-encoding", action="store_true")
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
    if args.model == "sam2_memory":
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
    args.protocol = args.protocol or ("davis" if args.dataset == "davis" else "diagnostic")
    args.prompt_mode = args.prompt_mode or ("mask" if args.protocol == "davis" else "point")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    model_options = {}
    for argument, option in ((args.backbone, "backbone"), (args.memory_bank_size, "memory_bank_size")):
        if argument is not None:
            model_options[option] = argument
    for disabled, option in ((args.no_presence_head, "presence_head"),
                             (args.no_object_pointers, "object_pointers"),
                             (args.no_temporal_position_encoding, "temporal_position_encoding")):
        if disabled:
            model_options[option] = False
    if model_options and args.model != "sam2_memory":
        parser.error("backbone and ablation options require --model sam2_memory")

    try:
        sequences = load_sequences(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nDataset configuration error:\n  {exc}\n", file=sys.stderr)
        return 3
    if not sequences:
        print("no sequences to evaluate")
        return 1

    try:
        evaluation = evaluate_tracker(
            args.model,
            sequences,
            object_index=args.object_index if args.object_index is not None or args.dataset == "davis" else 0,
            direction=Direction(args.direction),
            bidirectional=args.bidirectional,
            device=args.device,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            prompt_frame=args.prompt_frame,
            prompt_mode=args.prompt_mode,
            protocol=args.protocol,
            model_options=model_options,
            reuse_tracker=True,
            progress=not args.quiet,
        )
    except NotImplementedError as exc:
        # The stub raises this on purpose. Say so plainly instead of printing zeros.
        print(f"\n{args.model} is registered but not implemented yet:\n  {exc}\n")
        return 2
    except FileNotFoundError as exc:
        print(f"\n{args.model} needs a checkpoint:\n  {exc}\n")
        return 3
    except (ValueError, KeyError) as exc:
        print(f"\nEvaluation configuration error:\n  {exc}\n", file=sys.stderr)
        return 4

    print(evaluation.report())
    evaluation.config.update({"split": args.split if args.dataset == "davis" else None,
                              "resolution": args.resolution if args.dataset == "davis" else None,
                              "sequence_names": getattr(sequences, "names", [s.name for s in sequences] if isinstance(sequences, list) else []),
                              "sequence_limit": args.limit,
                              "is_full_split": args.limit is None})
    import platform
    import subprocess
    evaluation.config["python_version"] = platform.python_version()
    try:
        evaluation.config["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        evaluation.config["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        evaluation.config["git_revision"] = None
    if args.model == "sam2_memory":
        import transformers
        evaluation.config["torch_version"] = torch.__version__
        evaluation.config["transformers_version"] = transformers.__version__
        evaluation.config["gpu"] = torch.cuda.get_device_name() if args.device == "cuda" else None

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
