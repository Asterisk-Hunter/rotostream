"""Prove a tracker only reads the past.

A memory-attention tracker can silently attend to frames it has not visited yet -
in the paper's setting, the frame after the one being predicted - and the result
still *looks* fine: masks get sharper, occlusion handling improves, and the
benchmark number quietly becomes meaningless because the model is peeking. It is
also the single easiest bug to introduce when you build a memory bank, because a
sliding window over the whole clip is the natural thing to write.

The check here is deliberately blunt:

1. Run the tracker normally over a sequence and record the mask at frame ``t``.
2. Rerun it with every frame on the *unvisited* side of ``t`` replaced by random
   noise - for ``FORWARD``, all frames after ``t``; for ``BACKWARD``, all before.
3. Demand the mask at ``t`` is unchanged.

A causal tracker cannot tell the difference, because it never looked at those
frames. Anything else is leakage.

Noise rather than truncation is used on purpose: truncation only catches a model
that *indexes* into the future, whereas noise also catches one that reads a count
or aggregates over a window it should not have.

``api/tests/test_contract.py`` runs the same idea against the reference tracker.
This module is the version you point at your own model, on sequences where the
right answer is known.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.models import registry
from app.models.base import Direction, PromptSet, VideoObjectTracker

from .synthetic import ToySequence, available as available_sequences, build

__all__ = [
    "LeakCheck",
    "LeakReport",
    "check_causality",
    "factory_for",
    "DEFAULT_MAX_CHECKS",
]

#: How many frames to probe by default. Each probe costs a full tracker run, so
#: long sequences are sampled rather than exhaustively checked.
DEFAULT_MAX_CHECKS = 12


@dataclass(frozen=True)
class LeakCheck:
    """One poisoned-future probe at a single frame."""

    frame_index: int
    differing_pixels: int
    mask_area: int
    total_pixels: int

    def leaked(self, tolerance_pixels: int) -> bool:
        return self.differing_pixels > tolerance_pixels

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "differing_pixels": self.differing_pixels,
            "mask_area": self.mask_area,
            "total_pixels": self.total_pixels,
        }


@dataclass
class LeakReport:
    """Outcome of a causality run."""

    model: str
    sequence: str
    direction: Direction
    tolerance_pixels: int
    checks: list[LeakCheck] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def leaks(self) -> list[LeakCheck]:
        return [c for c in self.checks if c.leaked(self.tolerance_pixels)]

    @property
    def ok(self) -> bool:
        """True only when probes actually ran and all of them passed.

        A skipped report is deliberately NOT ok. Reporting "passed" for a model
        that was never exercised is the one failure mode this tool cannot have -
        it would hand you a clean bill of health on an untested memory bank.
        """
        return not self.leaks and not self.skipped and bool(self.checks)

    @property
    def worst(self) -> LeakCheck | None:
        return max(self.checks, key=lambda c: c.differing_pixels, default=None)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "sequence": self.sequence,
            "direction": self.direction.value,
            "tolerance_pixels": self.tolerance_pixels,
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "leaks": [c.to_dict() for c in self.leaks],
            "skipped": self.skipped,
        }

    def summary(self) -> str:
        head = (
            f"{self.model} on {self.sequence} [{self.direction.value}] - "
            f"{len(self.checks)} probe(s)"
        )
        if self.skipped:
            return (
                f"{head}\n  SKIPPED - NOT a pass, nothing was verified: "
                f"{'; '.join(self.skipped)}"
            )
        if not self.checks:
            return f"{head}\n  SKIPPED - NOT a pass: no frames could be probed"
        if self.ok:
            worst = self.worst
            detail = (
                f" (largest deviation {worst.differing_pixels}px at frame {worst.frame_index})"
                if worst and worst.differing_pixels
                else ""
            )
            return f"{head}\n  PASS - masks did not depend on unvisited frames{detail}"
        lines = [f"{head}", "  FAIL - the mask at these frames changed when future frames were corrupted:"]
        for check in self.leaks:
            lines.append(
                f"    frame {check.frame_index:4d}: {check.differing_pixels:6d} / "
                f"{check.total_pixels} px differ (mask area {check.mask_area})"
            )
        lines.append(
            "  The memory bank is reading the unvisited side. Check the queue bounds "
            "and the propagation order."
        )
        return "\n".join(lines)


def factory_for(model_key: str) -> Callable[[], VideoObjectTracker]:
    """A fresh tracker factory for a registered model key.

    A new instance per run is required: a checker that reuses one tracker would
    carry the memory bank across probes and measure nothing.
    """
    cls = registry.load_class(model_key)

    def make() -> VideoObjectTracker:
        return cls()

    return make


def _poison(frames: np.ndarray, start: int, stop: int, seed: int) -> np.ndarray:
    """Copy of ``frames`` with ``[start, stop)`` replaced by deterministic noise."""
    out = np.array(frames, copy=True)
    if start >= stop:
        return out
    rng = np.random.default_rng(seed)
    out[start:stop] = rng.integers(0, 256, size=out[start:stop].shape, dtype=np.uint8)
    return out


def _run_to(
    tracker: VideoObjectTracker,
    frames: np.ndarray,
    fps: float,
    prompt: PromptSet,
    direction: Direction,
    stop_index: int,
) -> dict[int, np.ndarray]:
    """Propagate from the prompt to ``stop_index`` and return every mask visited."""
    from app.models.frames import ArrayFrameSource

    tracker.set_video(ArrayFrameSource(frames, fps=fps))
    masks: dict[int, np.ndarray] = {prompt.frame_index: np.asarray(tracker.add_prompt(prompt).mask)}

    step = 1 if direction is Direction.FORWARD else -1
    index = prompt.frame_index + step
    limit = len(frames) if direction is Direction.FORWARD else -1
    while (step > 0 and index <= stop_index and index < limit) or (
        step < 0 and index >= stop_index >= 0
    ):
        masks[index] = np.asarray(tracker.propagate(index, direction).mask)
        index += step
    return masks


def _prompt_for(
    sequence: ToySequence,
    object_index: int,
    direction: Direction,
    prompt_frame: int | None,
) -> PromptSet:
    """Choose where to click, so both directions get a meaningful sweep.

    The first visible frame is the natural click for a forward sweep, but it makes
    every backward probe vacuous (nothing lies behind frame 0). For BACKWARD the
    prompt goes on the last visible frame instead, which is also what the API's
    bidirectional mode does when the user clicks late in a clip.
    """
    if prompt_frame is not None:
        return sequence.prompt_for(object_index, prompt_frame)
    if direction is Direction.BACKWARD:
        visible = np.nonzero(sequence.visible(object_index))[0]
        if len(visible):
            return sequence.prompt_for(object_index, int(visible[-1]))
    return sequence.prompt_for(object_index)


def _checkpoints(
    prompt_index: int, n_frames: int, direction: Direction, requested: list[int] | None, max_checks: int
) -> list[int]:
    """Frames reachable in ``direction`` from the prompt, sampled to ``max_checks``."""
    if direction is Direction.FORWARD:
        reachable = list(range(prompt_index, n_frames))
    else:
        reachable = list(range(prompt_index, -1, -1))
    if requested is not None:
        return [i for i in requested if i in set(reachable)]
    if len(reachable) <= max_checks:
        return reachable
    # Evenly spaced, always including both ends of the propagation.
    picks = np.linspace(0, len(reachable) - 1, max_checks).round().astype(int)
    return [reachable[i] for i in sorted(set(int(p) for p in picks))]


def check_causality(
    model: str | Callable[[], VideoObjectTracker],
    sequence: ToySequence,
    *,
    direction: Direction = Direction.FORWARD,
    object_index: int = 0,
    frames: list[int] | None = None,
    tolerance_pixels: int = 0,
    max_checks: int = DEFAULT_MAX_CHECKS,
    device: str = "cpu",
    checkpoint: str | None = None,
    prompt_frame: int | None = None,
    seed: int = 1234,
    preloaded_tracker: VideoObjectTracker | None = None,
) -> LeakReport:
    """Verify ``model`` only attends to already-visited frames.

    ``tolerance_pixels`` exists for floating-point nondeterminism in a real
    network (a handful of boundary pixels may legitimately flip), but the default
    is exact equality - a correct causal tracker is bit-identical here, because
    the poisoned frames are literally never read.
    """
    make_tracker = factory_for(model) if isinstance(model, str) else model
    label = model if isinstance(model, str) else getattr(make_tracker, "__name__", "factory")

    report = LeakReport(
        model=str(label), sequence=sequence.name, direction=direction, tolerance_pixels=tolerance_pixels
    )

    prompt = _prompt_for(sequence, object_index, direction, prompt_frame)
    probes = _checkpoints(prompt.frame_index, sequence.n_frames, direction, frames, max_checks)
    if not probes:
        report.skipped.append("no frames reachable in that direction from the prompt")
        return report

    def load() -> VideoObjectTracker:
        if preloaded_tracker is not None:
            # _run_to always calls set_video(), which resets bank and encoder cache.
            return preloaded_tracker
        tracker = make_tracker()
        try:
            tracker.load(device=device, checkpoint=checkpoint)
        except NotImplementedError as exc:
            report.skipped.append(f"model is not implemented: {exc}")
            raise
        return tracker

    try:
        reference = _run_to(
            load(), sequence.frames, sequence.fps, prompt, direction, probes[-1]
        )
    except NotImplementedError:
        return report

    for probe in probes:
        poisoned = (
            _poison(sequence.frames, probe + 1, sequence.n_frames, seed)
            if direction is Direction.FORWARD
            else _poison(sequence.frames, 0, probe, seed)
        )
        masks = _run_to(load(), poisoned, sequence.fps, prompt, direction, probe)

        expected = reference.get(probe)
        actual = masks.get(probe)
        if expected is None or actual is None:
            report.skipped.append(f"frame {probe} was never reached")
            continue

        report.checks.append(
            LeakCheck(
                frame_index=probe,
                differing_pixels=int(np.count_nonzero(expected != actual)),
                mask_area=int(np.count_nonzero(expected)),
                total_pixels=int(expected.size),
            )
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a tracker for future leakage.")
    parser.add_argument("--model", default="naive", help="registered tracker key")
    parser.add_argument("--sequence", default="occlusion", help="synthetic scenario name")
    parser.add_argument(
        "--direction", default="forward", choices=[d.value for d in Direction]
    )
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="explicit frames")
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=None,
        help="where to click (default: first visible frame forward, last backward)",
    )
    parser.add_argument("--max-checks", type=int, default=DEFAULT_MAX_CHECKS)
    parser.add_argument("--tolerance", type=int, default=0, help="allowed differing pixels")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32",
                        help="CUDA autocast precision; recorded in the JSON evidence")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--all", action="store_true", help="every scenario, both directions")
    parser.add_argument("--json", default=None, help="write probe evidence as JSON")
    args = parser.parse_args(argv)
    if args.precision == "bfloat16" and not args.device.startswith("cuda"):
        parser.error("--precision bfloat16 requires --device cuda")
    preloaded = None
    if args.model == "sam2_memory":
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        preloaded = registry.create(args.model)
        preloaded.load(device=args.device, checkpoint=args.checkpoint)

    scenarios = available_sequences() if args.all else [args.sequence]
    directions = [Direction.FORWARD, Direction.BACKWARD] if args.all else [Direction(args.direction)]

    failures = 0
    skipped = 0
    reports = []
    for name in scenarios:
        sequence = build(name)
        for direction in directions:
            try:
                if args.precision == "bfloat16":
                    import torch
                with (torch.autocast("cuda", dtype=torch.bfloat16)
                      if args.precision == "bfloat16" else nullcontext()):
                    report = check_causality(
                        args.model,
                        sequence,
                        direction=direction,
                        object_index=args.object_index,
                        frames=args.frames,
                        tolerance_pixels=args.tolerance,
                        max_checks=args.max_checks,
                        device=args.device,
                        checkpoint=args.checkpoint,
                        prompt_frame=args.prompt_frame,
                        preloaded_tracker=preloaded,
                    )
            except Exception as exc:  # noqa: BLE001 - report tooling failures per scenario
                print(f"{args.model} on {name} [{direction.value}] - ERROR: {type(exc).__name__}: {exc}")
                failures += 1
                continue
            print(report.summary())
            reports.append(report.to_dict())
            if report.skipped and not report.checks:
                skipped += 1
            else:
                failures += 0 if report.ok else 1
    print()
    if args.json:
        import json
        from pathlib import Path
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": args.model, "device": args.device,
                                    "precision": args.precision,
                                    "tolerance_pixels": args.tolerance, "max_checks": args.max_checks,
                                    "passed": not failures and not skipped, "reports": reports}, indent=2),
                        encoding="utf-8")
    if failures:
        print(f"{failures} causality check(s) FAILED.")
    if skipped:
        print(
            f"{skipped} causality check(s) could not run and proved nothing. "
            "Implement load() first."
        )
    if failures or skipped:
        return 1
    print("All causality checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
