"""The causality checker must be able to FAIL.

A checker that passes everything is worse than no checker, because it launders
leakage into a benchmark number you then quote. So the central tests here are a
tracker that deliberately reads the next frame (and must be caught) and the same
tracker made causal (and must pass).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.models.base import Direction, FrameResult, PromptSet, TrackerInfo, VideoObjectTracker
from rotostream_ml.leakcheck import LeakReport, check_causality
from rotostream_ml.synthetic import build


class PeekingTracker(VideoObjectTracker):
    """Reads frame ``index + 1`` when asked about ``index``. Pure future leakage."""

    key = "peeking"

    def __init__(self, peek: int = 1, threshold: int = 100) -> None:
        self.peek = peek
        self.threshold = threshold
        self._frames = None

    @classmethod
    def info(cls) -> TrackerInfo:
        return TrackerInfo(name=cls.key, description="deliberately leaky", implemented=True)

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        self._frames = None

    def set_video(self, frames) -> None:  # noqa: ANN001
        self._frames = frames

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        return self._mask(prompts.frame_index)

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        return self._mask(frame_index)

    def _mask(self, index: int) -> FrameResult:
        # A causal tracker would read `index`. This one looks ahead.
        source = index if self.peek == 0 else min(index + self.peek, len(self._frames) - 1)
        frame = self._frames[source]
        return FrameResult(mask=(frame[:, :, 0].astype(int) > self.threshold))


def _causal_factory() -> VideoObjectTracker:
    return PeekingTracker(peek=0)


def _leaky_factory() -> VideoObjectTracker:
    return PeekingTracker(peek=1)


class UnimplementedTracker(VideoObjectTracker):
    key = "unimplemented"

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        raise NotImplementedError("not built yet")

    def set_video(self, frames) -> None:  # noqa: ANN001
        pass

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        raise NotImplementedError

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        raise NotImplementedError


# ------------------------------------------------------------------- catching leaks
def test_a_tracker_that_reads_the_next_frame_is_caught():
    sequence = build("linear")
    report = check_causality(_leaky_factory, sequence, direction=Direction.FORWARD)

    assert not report.ok, "leakage went undetected"
    assert report.leaks, "a leaky tracker must produce leak entries"
    # Every probed frame except the last has a real future to depend on.
    assert len(report.leaks) >= len(report.checks) - 1
    assert all(check.differing_pixels > 0 for check in report.leaks)


def test_leak_is_caught_on_a_sequence_with_occlusion_too():
    report = check_causality(_leaky_factory, build("occlusion"))
    assert not report.ok
    assert max(check.differing_pixels for check in report.checks) > 0


def test_a_causal_tracker_passes():
    report = check_causality(_causal_factory, build("linear"), direction=Direction.FORWARD)

    assert report.ok, report.summary()
    assert report.checks and not report.leaks
    assert all(check.differing_pixels == 0 for check in report.checks)


def test_a_tracker_reading_the_previous_frame_is_caught_backwards():
    """Symmetric bug: memory attending forward while sweeping backwards."""
    sequence = build("linear")
    report = check_causality(
        lambda: PeekingTracker(peek=-1), sequence, direction=Direction.BACKWARD
    )
    assert not report.ok, "a backward sweep must catch reads of earlier frames"


def test_backward_probing_uses_a_late_prompt_so_it_is_not_vacuous():
    """A prompt on frame 0 would leave nothing behind it to probe."""
    sequence = build("linear")
    report = check_causality(_causal_factory, sequence, direction=Direction.BACKWARD)
    assert len(report.checks) > 1, "backward check probed nothing"


# --------------------------------------------------------------------- honesty
def test_an_unimplemented_model_reports_skipped_and_not_ok():
    """Never claim a pass for a model that was never exercised."""
    report = check_causality(UnimplementedTracker, build("linear"))

    assert not report.ok
    assert report.skipped
    assert "not implemented" in report.skipped[0]
    assert "NOT a pass" in report.summary()
    assert report.checks == []


def test_report_serialises_to_json_friendly_structures():
    report = check_causality(_causal_factory, build("linear"))
    payload = report.to_dict()

    assert payload["ok"] is True
    assert payload["model"] == "factory" or isinstance(payload["model"], str)
    assert payload["direction"] == "forward"
    assert isinstance(payload["checks"], list)


def test_empty_checks_is_not_a_pass():
    report = LeakReport(model="x", sequence="y", direction=Direction.FORWARD, tolerance_pixels=0)
    assert not report.ok
    assert "NOT a pass" in report.summary() or "SKIPPED" in report.summary()


# --------------------------------------------------------------------- plumbing
def test_explicit_frames_restrict_the_probe_set():
    sequence = build("linear")
    report = check_causality(_causal_factory, sequence, frames=[2, 5, 9])
    assert [check.frame_index for check in report.checks] == [2, 5, 9]


def test_max_checks_bounds_the_number_of_runs():
    sequence = build("occlusion")
    report = check_causality(_causal_factory, sequence, max_checks=4)
    assert len(report.checks) <= 4
    assert len(report.checks) >= 2


def test_tolerance_allows_a_small_number_of_differing_pixels():
    """Real networks are not bit-reproducible; the tolerance is the escape hatch."""
    sequence = build("linear")
    strict = check_causality(_leaky_factory, sequence, tolerance_pixels=0)
    lenient = check_causality(_leaky_factory, sequence, tolerance_pixels=10_000)

    assert not strict.ok
    assert lenient.ok, "a huge tolerance is expected to mask the difference"


def test_each_probe_uses_a_fresh_tracker_instance():
    """Chatty state across probes would make the whole check meaningless."""
    created: list[int] = []

    def counting_factory() -> VideoObjectTracker:
        created.append(1)
        return PeekingTracker(peek=0)

    sequence = build("linear")
    report = check_causality(counting_factory, sequence, frames=[1, 2, 3])
    # one reference run plus one per probe
    assert len(created) == len(report.checks) + 1


@pytest.mark.parametrize("name", ["linear", "occlusion", "reentry"])
def test_reference_tracker_is_causal_on_every_scenario(name):
    from app.models.naive import NaiveColorTracker

    report = check_causality(NaiveColorTracker, build(name), max_checks=6)
    assert report.ok, report.summary()
