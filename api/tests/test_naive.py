"""Behavioural tests for the reference baseline.

These double as executable documentation of the failure modes the memory-attention
tracker is supposed to fix: a frozen seed colour, and no occlusion head.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.models.base import ContractError, Direction, PointPrompt, PromptSet
from app.models.naive import NaiveColorTracker
from synthetic import OBJECT, ToySequence, iou, sliding_square


def build(sequence) -> NaiveColorTracker:
    tracker = NaiveColorTracker()
    tracker.load(device="cpu")
    tracker.set_video(sequence.source())
    return tracker


def prompt(sequence, x: float = 25.0, frame_index: int = 0, positive: bool = True) -> PromptSet:
    return PromptSet(
        frame_index=frame_index,
        points=(PointPrompt(x=x, y=sequence.centre_y, positive=positive),),
    )


def test_segments_a_solid_square():
    sequence = sliding_square(n_frames=6)
    result = build(sequence).add_prompt(prompt(sequence))
    assert iou(result.mask, sequence.masks[0]) >= 0.9


def test_info_reports_no_memory_and_no_training():
    info = NaiveColorTracker.info()
    assert info.uses_memory is False
    assert info.trainable is False
    assert info.implemented is True


def test_memory_state_is_honest_about_having_no_bank():
    state = build(sliding_square()).memory_state()
    assert state["bank_size"] == 0
    assert state["frames"] == []


def test_absent_object_yields_an_empty_mask_and_reports_absence():
    """Once the seeded colour is gone the baseline must report absence.

    Note that seeding on a *uniform* region legitimately returns that whole region
    (every pixel matches), so absence can only be observed after the colour vanishes
    - which is the case constructed here.
    """
    sequence = sliding_square(n_frames=4, half=8, x_start=20, x_end=20)
    frames = sequence.frames.copy()
    frames[1:] = (28, 30, 38)  # the object leaves after the prompted frame

    tracker = build(ToySequence(frames, sequence.masks))
    tracker.add_prompt(prompt(sequence, x=20.0))

    for index in range(1, sequence.n_frames):
        result = tracker.propagate(index, Direction.FORWARD)
        assert not result.mask.any(), f"frame {index}: expected no colour match"
        assert result.object_present is False


def test_negative_clicks_remove_a_same_coloured_distractor():
    """A second blob of the identical colour must be excluded by a negative click."""
    sequence = sliding_square(n_frames=1, size=(96, 128), half=8, x_start=20, x_end=20)
    frames = sequence.frames.copy()
    frames[0, 66:82, 96:112] = OBJECT  # distractor blob of the SAME colour
    tracker = build(ToySequence(frames, sequence.masks))

    prompts = PromptSet(
        frame_index=0,
        points=(
            PointPrompt(x=20, y=sequence.centre_y, positive=True),
            PointPrompt(x=104, y=74, positive=False),
        ),
    )
    mask = tracker.add_prompt(prompts).mask
    assert mask[sequence.centre_y, 20], "the prompted object should be in the mask"
    assert not mask[74, 104], "the negatively prompted distractor must be excluded"


def test_propagate_before_prompt_is_a_contract_error():
    with pytest.raises(ContractError, match="add_prompt"):
        build(sliding_square()).propagate(1, Direction.FORWARD)


def test_add_prompt_without_a_positive_point_is_a_contract_error():
    sequence = sliding_square(n_frames=2)
    tracker = build(sequence)
    with pytest.raises(ContractError, match="positive point"):
        tracker.add_prompt(
            PromptSet(frame_index=0, points=(PointPrompt(x=20, y=48, positive=False),))
        )


def test_box_prompt_is_accepted_as_a_fallback_seed():
    from app.models.base import BoxPrompt

    sequence = sliding_square(n_frames=2)
    tracker = build(sequence)
    result = tracker.add_prompt(
        PromptSet(frame_index=0, box=BoxPrompt(x0=8, y0=36, x1=34, y1=60))
    )
    assert iou(result.mask, sequence.masks[0]) >= 0.5


@pytest.mark.xfail(
    reason="documents the baseline's weakness: with no memory the seeded colour is "
    "frozen, so a colour change drops the object",
    strict=True,
)
def test_frozen_seed_cannot_follow_a_colour_change():
    sequence = sliding_square(n_frames=8)
    frames = sequence.frames.copy()
    for t in range(4, sequence.n_frames):
        frames[t][sequence.masks[t]] = (60, 90, 220)  # object turns blue

    tracker = build(ToySequence(frames, sequence.masks))
    tracker.add_prompt(prompt(sequence))

    tracked = [
        iou(tracker.propagate(t, Direction.FORWARD).mask, sequence.masks[t])
        for t in range(4, sequence.n_frames)
    ]
    # If the baseline actually held the object through the colour change this would
    # pass; it does not, which is precisely why the memory stack exists.
    assert min(tracked) >= 0.5, f"baseline lost the object after the colour change: {tracked}"


def test_baseline_tracks_colour_not_identity():
    """The core failure the memory stack exists to fix.

    With no memory bank and no occlusion head the baseline cannot tell "the tracked
    object left the scene" apart from "a different object of a similar colour
    appeared", so it adopts the impostor as if it were the tracked object.
    """
    sequence = sliding_square(n_frames=6, half=10, x_start=20, x_end=20)
    frames = sequence.frames.copy()
    frames[3:] = (28, 30, 38)  # the tracked object leaves the scene
    frames[3:, 60:80, 96:116] = OBJECT  # an unrelated same-coloured blob appears

    tracker = build(ToySequence(frames, sequence.masks))
    tracker.add_prompt(prompt(sequence, x=20.0))
    result = tracker.propagate(5, Direction.FORWARD)

    assert result.mask[70, 106], "the baseline adopts the impostor as the tracked object"
    assert result.object_present is True, "and reports it as present, with confidence"
