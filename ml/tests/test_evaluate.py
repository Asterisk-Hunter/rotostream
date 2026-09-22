"""Evaluator tests.

The important one is the *known answer*: a tracker that returns the ground truth
perfectly must score J = 1 and F = 1. If that ever fails, the harness is lying
about every other number too. The presence-head diagnostics get the same
treatment, because they are what tells you whether the occlusion head works.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.models.base import Direction, FrameResult, PromptSet, TrackerInfo, VideoObjectTracker
from rotostream_ml.evaluate import PresenceStats, evaluate_tracker
from rotostream_ml.synthetic import OBJECT_COLOURS, build


class ExactColourTracker(VideoObjectTracker):
    """Masks the pixels matching the object colour. An oracle on clean scenarios."""

    key = "exact_colour"

    def __init__(self, colour=OBJECT_COLOURS[0]) -> None:
        self.colour = np.array(colour, dtype=np.uint8)
        self._frames = None

    @classmethod
    def info(cls) -> TrackerInfo:
        return TrackerInfo(name=cls.key, implemented=True)

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        self._frames = None

    def set_video(self, frames) -> None:  # noqa: ANN001
        self._frames = frames

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        return self.propagate(prompts.frame_index, Direction.FORWARD)

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        frame = self._frames[frame_index]
        mask = np.all(frame == self.colour, axis=-1)
        return FrameResult(mask=mask, score=1.0, object_present=bool(mask.any()))


class AlwaysPresentEmptyTracker(VideoObjectTracker):
    """Claims the object is always there but never draws anything.

    The interesting case: J on absent frames is 1.0 (empty vs empty) while the
    presence flag is wrong, which is exactly why the report shows both.
    """

    key = "always_present_empty"

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        self._shape = None

    def set_video(self, frames) -> None:  # noqa: ANN001
        self._shape = frames.shape

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        return FrameResult(mask=np.zeros(self._shape, dtype=bool), object_present=True)

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        return FrameResult(mask=np.zeros(self._shape, dtype=bool), object_present=True)


class UnimplementedTracker(VideoObjectTracker):
    key = "unimplemented_eval"

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        raise NotImplementedError("not built yet")

    def set_video(self, frames) -> None:  # noqa: ANN001
        pass

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        raise NotImplementedError

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        raise NotImplementedError


# ----------------------------------------------------------------- known answers
def test_a_perfect_tracker_scores_one():
    sequence = build("linear")
    evaluation = evaluate_tracker(ExactColourTracker, [sequence], progress=False)

    assert evaluation.score.j == pytest.approx(1.0)
    assert evaluation.score.f == pytest.approx(1.0)
    assert evaluation.score.jf == pytest.approx(1.0)
    assert evaluation.presence.accuracy == pytest.approx(1.0)


def test_a_perfect_tracker_scores_one_on_the_hard_scenarios_too():
    """Colour matching is an oracle even through occlusion and re-entry."""
    sequences = [build("occlusion"), build("reentry")]
    evaluation = evaluate_tracker(ExactColourTracker, sequences, progress=False)

    assert evaluation.score.j == pytest.approx(1.0)
    assert evaluation.score.f == pytest.approx(1.0)


def test_scoring_covers_every_frame_including_before_and_after_a_prompt():
    sequence = build("occlusion")
    evaluation = evaluate_tracker(
        ExactColourTracker, [sequence], prompt_frame=10, bidirectional=True, progress=False
    )
    assert evaluation.sequences[0].score.n_frames == sequence.n_frames


# ------------------------------------------------------------------ presence
def test_presence_stats_are_computed_from_the_trackers_own_flag():
    sequence = build("occlusion")
    evaluation = evaluate_tracker(ExactColourTracker, [sequence], progress=False)
    item = evaluation.sequences[0]

    assert item.presence.accuracy == pytest.approx(1.0)
    assert item.presence.fp == 0 and item.presence.fn == 0
    assert item.presence.tp == int(sequence.visible(0).sum())
    assert item.presence.tn == item.n_absent


def test_a_tracker_that_never_reports_absence_is_caught_by_the_diagnostics():
    sequence = build("occlusion")
    evaluation = evaluate_tracker(AlwaysPresentEmptyTracker, [sequence], progress=False)
    item = evaluation.sequences[0]

    assert item.n_absent > 0
    # Empty prediction against empty ground truth still scores 1.0 by convention...
    assert item.j_absent == pytest.approx(1.0)
    # ...so the presence flag is the only thing that reveals the failure. Claiming
    # "present" everywhere gives perfect recall and one false positive per absent
    # frame, which is what the report surfaces.
    assert item.presence.fp == item.n_absent
    assert item.presence.fn == 0
    assert item.presence.recall == pytest.approx(1.0)
    assert item.presence.precision < 1.0
    assert item.presence.accuracy == pytest.approx(item.n_visible / sequence.n_frames)


def test_j_is_split_by_whether_the_object_was_on_screen():
    sequence = build("occlusion")
    evaluation = evaluate_tracker(AlwaysPresentEmptyTracker, [sequence], progress=False)
    item = evaluation.sequences[0]

    assert item.n_visible + item.n_absent == sequence.n_frames
    assert item.j_visible == pytest.approx(0.0)  # never drew the object
    assert item.j_absent == pytest.approx(1.0)  # correctly empty there


def test_presence_stats_arithmetic():
    stats = PresenceStats()
    for predicted, actual in [(True, True), (True, False), (False, True), (False, False)]:
        stats.add(predicted, actual)

    assert (stats.tp, stats.fp, stats.fn, stats.tn) == (1, 1, 1, 1)
    assert stats.accuracy == pytest.approx(0.5)
    assert stats.precision == pytest.approx(0.5)
    assert stats.recall == pytest.approx(0.5)
    assert stats.f1 == pytest.approx(0.5)


def test_presence_stats_of_nothing_is_zero_not_a_crash():
    stats = PresenceStats()
    assert stats.total == 0
    assert stats.accuracy == 0.0 and stats.f1 == 0.0


# ------------------------------------------------------------------ plumbing
def test_multiple_sequences_aggregate_per_sequence():
    sequences = [build("linear"), build("reentry")]
    evaluation = evaluate_tracker(ExactColourTracker, sequences, progress=False)

    assert [item.name for item in evaluation.sequences] == ["linear", "reentry"]
    assert evaluation.score.n_frames == sum(item.score.n_frames for item in evaluation.sequences)
    assert evaluation.score.j == pytest.approx(1.0)


def test_report_and_serialisation_work():
    evaluation = evaluate_tracker(ExactColourTracker, [build("linear")], progress=False)
    payload = evaluation.to_dict()

    assert payload["j"] == pytest.approx(1.0)
    assert payload["dataset"] == "synthetic"
    assert len(payload["sequences"]) == 1
    assert "presence" in payload
    assert "sequence" in evaluation.report()
    assert evaluation.csv_rows()[0]["sequence"] == "linear"


def test_an_unimplemented_model_raises_rather_than_scoring_zero():
    with pytest.raises(NotImplementedError):
        evaluate_tracker(UnimplementedTracker, [build("linear")], progress=False)


def test_an_unimplemented_model_is_rejected_before_any_sequence_runs():
    """Better to fail once up front than to print a table of zeros."""
    class CountingTracker(UnimplementedTracker):
        def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
            raise NotImplementedError("nope")

    with pytest.raises(NotImplementedError):
        evaluate_tracker(CountingTracker, [build("linear"), build("occlusion")], progress=False)


def test_causality_and_evaluation_agree_on_a_fresh_instance_per_run():
    """Shared state between sequences would silently carry a memory bank over."""
    instances: list[int] = []

    def factory() -> VideoObjectTracker:
        instances.append(1)
        return ExactColourTracker()

    sequences = [build("linear"), build("occlusion"), build("reentry")]
    evaluate_tracker(factory, sequences, progress=False)
    assert len(instances) == len(sequences)


def test_object_index_selects_which_object_is_scored():
    sequence = build("distractor")
    assert sequence.n_objects == 2

    for object_index in (0, 1):
        evaluation = evaluate_tracker(
            ExactColourTracker, [sequence], object_index=object_index, progress=False
        )
        # The colour oracle draws BOTH objects, so scoring one of them against the
        # other's ground truth must be clearly worse than scoring its own set.
        assert evaluation.score.j < 1.0
        assert evaluation.sequences[0].n_objects == 2
