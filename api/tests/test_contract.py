"""Contract tests every registered tracker must pass.

Parametrised over the registry: implement your model, flip ``implemented=True`` in
``info()``, and it is covered by these tests automatically. Run with::

    .venv/Scripts/python -m pytest api/tests/test_contract.py -v

The future-leakage tests are the important ones. They compare two runs whose
frames are byte-identical up to the point being predicted but differ *after* it.
A causal tracker produces identical masks; a tracker whose memory attends to the
future places its mask on the second variant's object instead, and fails loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.models import registry
from app.models.base import (
    Direction,
    FrameResult,
    PointPrompt,
    PromptSet,
    check_frame_result,
)
from synthetic import iou, occluded_square, sliding_square, teleporting_square

IMPLEMENTED = [info.name for info in registry.available() if info.implemented]

pytestmark = pytest.mark.skipif(
    not IMPLEMENTED, reason="no implemented trackers registered"
)

LEAK_TOLERANCE = 0.99


# --------------------------------------------------------------------- helpers
def new_tracker(key: str, sequence):
    tracker = registry.create(key)
    tracker.load(device="cpu")
    tracker.set_video(sequence.source())
    return tracker


def prompt_at(sequence, frame_index: int, x: float, y: float | None = None) -> PromptSet:
    return PromptSet(
        frame_index=frame_index,
        points=(PointPrompt(x=x, y=sequence.centre_y if y is None else y, positive=True),),
    )


def walk(key: str, sequence, prompt_frame: int, x: float, indices, direction):
    """Prompt, then visit ``indices`` in order, returning the masks."""
    tracker = new_tracker(key, sequence)
    tracker.add_prompt(prompt_at(sequence, prompt_frame, x))
    return [tracker.propagate(index, direction).mask for index in indices]


# ------------------------------------------------------------------ basics
@pytest.mark.parametrize("key", IMPLEMENTED)
def test_prompt_then_forward_propagation(key):
    sequence = sliding_square(n_frames=8)
    height, width = sequence.size
    tracker = new_tracker(key, sequence)

    first = tracker.add_prompt(prompt_at(sequence, 0, 30))
    assert isinstance(first, FrameResult)
    check_frame_result(first, (height, width))
    assert first.mask.any(), (
        "prompting the centre of a solid square must produce a non-empty mask"
    )

    for index in range(1, sequence.n_frames):
        result = tracker.propagate(index, Direction.FORWARD)
        check_frame_result(result, (height, width))


@pytest.mark.parametrize("key", IMPLEMENTED)
def test_prompt_then_backward_propagation(key):
    sequence = sliding_square(n_frames=8)
    height, width = sequence.size
    tracker = new_tracker(key, sequence)
    last = sequence.n_frames - 1

    tracker.add_prompt(prompt_at(sequence, last, 30, sequence.centre_y))
    for index in range(last - 1, -1, -1):
        result = tracker.propagate(index, Direction.BACKWARD)
        check_frame_result(result, (height, width))


@pytest.mark.parametrize("key", IMPLEMENTED)
def test_tracks_the_object_it_can_see(key):
    """On an unobstructed clip the mask should stay on the square."""
    sequence = sliding_square(n_frames=8)
    tracker = new_tracker(key, sequence)
    tracker.add_prompt(prompt_at(sequence, 0, 25))

    scores = []
    for index in range(1, sequence.n_frames):
        mask = tracker.propagate(index, Direction.FORWARD).mask
        scores.append(iou(mask, sequence.masks[index]))
    assert min(scores) >= 0.5, f"mask drifted off the object: IoU per frame {scores}"


@pytest.mark.parametrize("key", IMPLEMENTED)
def test_absent_object_implies_empty_mask(key):
    """The occlusion head contract: object_present=False must mean an empty mask."""
    sequence = occluded_square(n_frames=12)
    tracker = new_tracker(key, sequence)
    tracker.add_prompt(prompt_at(sequence, 0, 24))

    for index in range(1, sequence.n_frames):
        result = tracker.propagate(index, Direction.FORWARD)
        if not result.object_present:
            assert not result.mask.any(), (
                f"frame {index}: object_present=False but the mask is non-empty"
            )


@pytest.mark.parametrize("key", IMPLEMENTED)
def test_reset_returns_to_a_clean_state(key):
    sequence = sliding_square(n_frames=6)
    tracker = new_tracker(key, sequence)
    tracker.add_prompt(prompt_at(sequence, 0, 25))
    tracker.propagate(1, Direction.FORWARD)
    tracker.reset()

    again = tracker.add_prompt(prompt_at(sequence, 0, 25)).mask
    fresh = new_tracker(key, sequence).add_prompt(prompt_at(sequence, 0, 25)).mask
    assert iou(again, fresh) >= LEAK_TOLERANCE, "reset() left state behind"


# ------------------------------------------------------------ future leakage
@pytest.mark.parametrize("key", IMPLEMENTED)
def test_no_future_leakage_forward(key):
    """Frames 0-3 are identical across both variants; only 4+ differs.

    Predicting frames 1-3 must therefore give the same answer in both runs.
    """
    shared_x = 20
    variant_a = teleporting_square(n_frames=8, switch_at=4, x_before=shared_x, x_after=100)
    variant_b = teleporting_square(n_frames=8, switch_at=4, x_before=shared_x, x_after=40)
    assert np.array_equal(variant_a.frames[:4], variant_b.frames[:4])
    assert not np.array_equal(variant_a.frames[4:], variant_b.frames[4:])

    indices = [1, 2, 3]
    masks_a = walk(key, variant_a, 0, shared_x, indices, Direction.FORWARD)
    masks_b = walk(key, variant_b, 0, shared_x, indices, Direction.FORWARD)

    for index, mask_a, mask_b in zip(indices, masks_a, masks_b):
        score = iou(mask_a, mask_b)
        assert score >= LEAK_TOLERANCE, (
            f"future leakage: forward prediction for frame {index} changed when only "
            f"frames 4+ changed (IoU {score:.4f}). Memory must only attend to frames "
            "already visited in the propagation direction."
        )


@pytest.mark.parametrize("key", IMPLEMENTED)
def test_no_future_leakage_backward(key):
    """Frames 3-7 are identical across both variants; only 0-2 differs.

    Walking backwards from frame 5, predictions for frames 4 and 3 must not change.
    """
    variant_a = teleporting_square(n_frames=8, switch_at=3, x_before=20, x_after=100)
    variant_b = teleporting_square(n_frames=8, switch_at=3, x_before=70, x_after=100)
    assert np.array_equal(variant_a.frames[3:], variant_b.frames[3:])
    assert not np.array_equal(variant_a.frames[:3], variant_b.frames[:3])

    indices = [4, 3]
    masks_a = walk(key, variant_a, 5, 100, indices, Direction.BACKWARD)
    masks_b = walk(key, variant_b, 5, 100, indices, Direction.BACKWARD)

    for index, mask_a, mask_b in zip(indices, masks_a, masks_b):
        score = iou(mask_a, mask_b)
        assert score >= LEAK_TOLERANCE, (
            f"future leakage: backward prediction for frame {index} changed when only "
            f"frames 0-2 changed (IoU {score:.4f})."
        )


@pytest.mark.parametrize("key", IMPLEMENTED)
def test_memory_bank_reports_no_future_frames(key):
    """Structural check on ``memory_state()`` when the plugin reports frame indices."""
    sequence = sliding_square(n_frames=8)
    tracker = new_tracker(key, sequence)
    tracker.add_prompt(prompt_at(sequence, 0, 25))

    for index in range(1, 5):
        tracker.propagate(index, Direction.FORWARD)
        state = tracker.memory_state()
        frames = state.get("frames") if isinstance(state, dict) else None
        if not frames:
            pytest.skip("memory_state() does not report frame indices")
        assert max(frames) <= index, (
            f"after propagating to frame {index} the memory bank holds {max(frames)}"
        )
