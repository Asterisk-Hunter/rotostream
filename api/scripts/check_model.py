#!/usr/bin/env python
"""Verify a registered tracker against the plugin contract.

    python api/scripts/check_model.py                # every implemented tracker
    python api/scripts/check_model.py sam2_memory    # one tracker, while iterating

Runs the same checks as ``api/tests/test_contract.py`` - shape and dtype rules,
forward/backward propagation, reset, and the future-leakage tests - but standalone
so you can point it at a model you are actively working on and get a readable
report instead of a pytest diff.

Exit code is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
for extra in (API_ROOT, API_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np  # noqa: E402

from app.models import registry  # noqa: E402
from app.models.base import (  # noqa: E402
    Direction,
    PointPrompt,
    PromptSet,
    check_frame_result,
)
from synthetic import iou, sliding_square, teleporting_square  # noqa: E402

LEAK_TOLERANCE = 0.99
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.passes = 0
        self.skips = 0

    def ok(self, label: str, detail: str = "") -> None:
        self.passes += 1
        suffix = f" {DIM}{detail}{RESET}" if detail else ""
        print(f"  {GREEN}PASS{RESET}  {label}{suffix}")

    def fail(self, label: str, detail: str) -> None:
        self.failures += 1
        print(f"  {RED}FAIL{RESET}  {label}\n        {detail}")

    def skip(self, label: str, detail: str = "") -> None:
        self.skips += 1
        print(f"  {YELLOW}SKIP{RESET}  {label}{(' ' + detail) if detail else ''}")


def prompt_at(sequence, frame_index: int, x: float) -> PromptSet:
    return PromptSet(
        frame_index=frame_index,
        points=(PointPrompt(x=x, y=sequence.centre_y, positive=True),),
    )


def fresh(key: str, sequence):
    tracker = registry.create(key)
    tracker.load(device="cpu")
    tracker.set_video(sequence.source())
    return tracker


def walk(key: str, sequence, prompt_frame: int, x: float, indices, direction):
    tracker = fresh(key, sequence)
    tracker.add_prompt(prompt_at(sequence, prompt_frame, x))
    return [tracker.propagate(index, direction).mask for index in indices]


def check_tracker(key: str, report: Report) -> None:
    print(f"\n{key}")
    print(f"{DIM}{'-' * (len(key) + 2)}{RESET}")

    # --- registration and metadata -------------------------------------------
    try:
        info = registry.load_class(key).info()
    except Exception as exc:  # noqa: BLE001
        report.fail("resolves from the registry", f"{type(exc).__name__}: {exc}")
        return
    report.ok("resolves from the registry", f"uses_memory={info.uses_memory} trainable={info.trainable}")

    if not info.implemented:
        report.skip("contract checks", f"info().implemented is False ({info.error or 'stub'})")
        return

    # --- prompt + propagate ---------------------------------------------------
    sequence = sliding_square(n_frames=8)
    height, width = sequence.size
    try:
        tracker = fresh(key, sequence)
        first = tracker.add_prompt(prompt_at(sequence, 0, 30))
        check_frame_result(first, (height, width))
        if not first.mask.any():
            report.fail("prompt produces a mask", "prompting a solid object returned an empty mask")
        else:
            report.ok("prompt produces a mask", f"{int(first.mask.sum())} px")
    except Exception as exc:  # noqa: BLE001
        report.fail("prompt produces a mask", f"{type(exc).__name__}: {exc}")
        return

    try:
        for index in range(1, sequence.n_frames):
            check_frame_result(tracker.propagate(index, Direction.FORWARD), (height, width))
        report.ok("forward propagation", f"{sequence.n_frames - 1} frames")
    except Exception as exc:  # noqa: BLE001
        report.fail("forward propagation", f"{type(exc).__name__}: {exc}")

    try:
        backwards = fresh(key, sequence)
        backwards.add_prompt(prompt_at(sequence, sequence.n_frames - 1, 30))
        for index in range(sequence.n_frames - 2, -1, -1):
            check_frame_result(backwards.propagate(index, Direction.BACKWARD), (height, width))
        report.ok("backward propagation", f"{sequence.n_frames - 1} frames")
    except Exception as exc:  # noqa: BLE001
        report.fail("backward propagation", f"{type(exc).__name__}: {exc}")

    # --- reset ---------------------------------------------------------------
    try:
        stateful = fresh(key, sequence)
        stateful.add_prompt(prompt_at(sequence, 0, 25))
        stateful.propagate(1, Direction.FORWARD)
        stateful.reset()
        again = stateful.add_prompt(prompt_at(sequence, 0, 25)).mask
        clean = fresh(key, sequence).add_prompt(prompt_at(sequence, 0, 25)).mask
        score = iou(again, clean)
        if score >= LEAK_TOLERANCE:
            report.ok("reset clears the memory bank", f"IoU {score:.4f}")
        else:
            report.fail("reset clears the memory bank", f"post-reset mask differs (IoU {score:.4f})")
    except Exception as exc:  # noqa: BLE001
        report.fail("reset clears the memory bank", f"{type(exc).__name__}: {exc}")

    # --- leakage -------------------------------------------------------------
    shared = 20
    variant_a = teleporting_square(n_frames=8, switch_at=4, x_before=shared, x_after=100)
    variant_b = teleporting_square(n_frames=8, switch_at=4, x_before=shared, x_after=40)
    try:
        indices = [1, 2, 3]
        masks_a = walk(key, variant_a, 0, shared, indices, Direction.FORWARD)
        masks_b = walk(key, variant_b, 0, shared, indices, Direction.FORWARD)
        worst = min(iou(a, b) for a, b in zip(masks_a, masks_b))
        if worst >= LEAK_TOLERANCE:
            report.ok("no future leakage (forward)", f"worst IoU {worst:.4f}")
        else:
            report.fail(
                "no future leakage (forward)",
                f"predictions changed when only frames 4+ changed (worst IoU {worst:.4f}). "
                "Memory must only attend to frames already visited in the propagation direction.",
            )
    except Exception as exc:  # noqa: BLE001
        report.fail("no future leakage (forward)", f"{type(exc).__name__}: {exc}")

    back_a = teleporting_square(n_frames=8, switch_at=3, x_before=20, x_after=100)
    back_b = teleporting_square(n_frames=8, switch_at=3, x_before=70, x_after=100)
    try:
        indices = [4, 3]
        masks_a = walk(key, back_a, 5, 100, indices, Direction.BACKWARD)
        masks_b = walk(key, back_b, 5, 100, indices, Direction.BACKWARD)
        worst = min(iou(a, b) for a, b in zip(masks_a, masks_b))
        if worst >= LEAK_TOLERANCE:
            report.ok("no future leakage (backward)", f"worst IoU {worst:.4f}")
        else:
            report.fail(
                "no future leakage (backward)",
                f"predictions changed when only frames 0-2 changed (worst IoU {worst:.4f})",
            )
    except Exception as exc:  # noqa: BLE001
        report.fail("no future leakage (backward)", f"{type(exc).__name__}: {exc}")

    # --- occlusion contract ---------------------------------------------------
    try:
        tracker = fresh(key, sliding_square(n_frames=6))
        tracker.add_prompt(prompt_at(sequence, 0, 30))
        for index in range(1, 6):
            result = tracker.propagate(index, Direction.FORWARD)
            if not result.object_present and result.mask.any():
                raise AssertionError(
                    f"frame {index}: object_present=False but the mask is non-empty"
                )
        report.ok("absent object => empty mask")
    except Exception as exc:  # noqa: BLE001
        report.fail("absent object => empty mask", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*", help="tracker keys to check (default: all implemented)")
    parser.add_argument("--all", action="store_true", help="include unimplemented stubs")
    args = parser.parse_args()

    keys = args.keys
    if not keys:
        keys = [info.name for info in registry.available() if info.implemented or args.all]
    if not keys:
        print("no trackers to check. Implement one in api/app/models/sam2_memory.py")
        return 1

    report = Report()
    for key in keys:
        try:
            check_tracker(key, report)
        except Exception:  # noqa: BLE001 - never let the tool itself crash
            report.fail(f"{key}: unexpected error", traceback.format_exc(limit=3))

    print(f"\n{report.passes} passed, {report.failures} failed, {report.skips} skipped")
    if report.failures:
        print(f"{RED}Contract not satisfied.{RESET} See api/app/models/base.py.")
    elif report.skips:
        print(
            f"{YELLOW}Nothing verified.{RESET} The marker is still a stub - flip "
            "implemented=True in info() once load() works."
        )
    else:
        print(f"{GREEN}Contract satisfied.{RESET}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
