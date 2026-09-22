"""Synthetic scenarios must be deterministic and hide what they claim to hide.

These clips are the measuring stick for a tracker, so the tests are about the
scenarios themselves being honest: occlusion must actually occlude, re-entry must
actually leave the frame, the distractor must actually be a hard case, and the
ground truth must line up with the pixels.
"""
from __future__ import annotations

import numpy as np
import pytest

from rotostream_ml import synthetic
from rotostream_ml.sequences import VideoSequence, interior_point

OBJECT_COLOUR = np.array(synthetic.OBJECT_COLOURS[0], dtype=np.uint8)


# ------------------------------------------------------------------- geometry
@pytest.mark.parametrize("name", synthetic.available())
def test_every_scenario_has_consistent_shapes(name):
    sequence = synthetic.build(name)

    assert sequence.frames is not None
    assert sequence.frames.ndim == 4 and sequence.frames.shape[-1] == 3
    assert sequence.masks.ndim == 4
    assert sequence.frames.shape[0] == sequence.masks.shape[0]
    assert sequence.frames.shape[1:3] == sequence.masks.shape[2:4]
    assert sequence.frames.dtype == np.uint8
    assert sequence.masks.dtype == bool
    assert len(sequence) == sequence.n_frames


@pytest.mark.parametrize("name", ["linear", "occlusion", "reentry"])
def test_ground_truth_matches_the_object_pixels(name):
    """The mask must be exactly where the object colour is - no off-by-one."""
    sequence = synthetic.build(name)
    for frame_index in range(sequence.n_frames):
        painted = np.all(sequence.frames[frame_index] == OBJECT_COLOUR, axis=-1)
        assert np.array_equal(sequence.masks[frame_index, 0], painted), (
            f"{name} frame {frame_index}: mask and pixels disagree"
        )


def test_distractor_pixels_cover_both_objects():
    """Both squares share one colour, so the pixels are their union, not one mask."""
    sequence = synthetic.build("distractor")
    for frame_index in range(sequence.n_frames):
        painted = np.all(sequence.frames[frame_index] == OBJECT_COLOUR, axis=-1)
        union = sequence.masks[frame_index, 0] | sequence.masks[frame_index, 1]
        assert np.array_equal(painted, union), f"frame {frame_index}"


def test_color_shift_masks_stay_exact_while_the_colour_moves():
    """The colour blends, so the mask is checked by geometry instead."""
    sequence = synthetic.build("color_shift")
    for frame_index in range(sequence.n_frames):
        mask = sequence.masks[frame_index, 0]
        pixels = sequence.frames[frame_index][mask]
        # One uniform colour inside, and the region is a solid rectangle.
        assert len(np.unique(pixels, axis=0)) == 1, f"frame {frame_index}: object not uniform"
        ys, xs = np.nonzero(mask)
        assert mask.sum() == (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        assert not np.all(pixels[0] == sequence.frames[frame_index][~mask][0])


@pytest.mark.parametrize("name", synthetic.available())
def test_builders_are_deterministic(name):
    first = synthetic.build(name)
    second = synthetic.build(name)
    assert np.array_equal(first.frames, second.frames)
    assert np.array_equal(first.masks, second.masks)


def test_unknown_scenario_lists_the_valid_ones():
    with pytest.raises(KeyError, match="occlusion"):
        synthetic.build("no_such_scenario")


# ------------------------------------------------------------------- occlusion
def test_occlusion_really_hides_the_object():
    sequence = synthetic.build("occlusion")
    hidden = [index for index in range(sequence.n_frames) if not sequence.visible(0)[index]]

    assert len(hidden) >= 3, "the occlusion scenario must occlude for several frames"
    assert hidden == sorted(hidden), "hidden frames should be one contiguous passage"
    for index in hidden:
        assert not sequence.masks[index, 0].any(), "a hidden object must have an empty mask"


def test_occlusion_is_partial_before_it_is_total():
    """The bar approaches, so some frames have a partly covered object."""
    sequence = synthetic.build("occlusion")
    areas = sequence.masks[:, 0].reshape(sequence.n_frames, -1).sum(axis=1)
    full = areas[0]
    partial = [area for area in areas if 0 < area < full]

    assert partial, "expected frames where the object is only partly covered"
    assert all(area < full for area in partial)


# --------------------------------------------------------------------- reentry
def test_reentry_leaves_the_frame_and_comes_back():
    sequence = synthetic.build("reentry")
    visible = sequence.visible(0)
    hidden = [index for index in range(sequence.n_frames) if not visible[index]]

    assert hidden, "the object must actually leave the frame"
    assert min(hidden) > 0, "the object should start on screen"
    assert max(hidden) < sequence.n_frames - 1, "the object must come back"
    # Contiguous absence, then re-appearing on the other side.
    assert hidden == list(range(min(hidden), max(hidden) + 1))


def test_reentry_gap_is_tunable():
    short = synthetic.build("reentry", gap=2)
    long = synthetic.build("reentry", gap=8)
    assert long.visible(0).sum() < short.visible(0).sum()


# ------------------------------------------------------------------ distractor
def test_distractor_is_two_identical_objects_that_never_overlap():
    sequence = synthetic.build("distractor")
    assert sequence.n_objects == 2
    assert not np.any(sequence.masks[:, 0] & sequence.masks[:, 1])

    # Same colour on purpose: only memory can tell them apart.
    for index in (0, sequence.n_frames // 2):
        assert np.all(sequence.frames[index][sequence.masks[index, 0]] == OBJECT_COLOUR)
        assert np.all(sequence.frames[index][sequence.masks[index, 1]] == OBJECT_COLOUR)


def test_distractor_objects_move_in_opposite_directions():
    sequence = synthetic.build("distractor")

    def centre_x(object_index: int, frame_index: int) -> float:
        ys, xs = np.nonzero(sequence.masks[frame_index, object_index])
        return float(xs.mean())

    assert centre_x(0, -1) > centre_x(0, 0)
    assert centre_x(1, -1) < centre_x(1, 0)


# ----------------------------------------------------------------- color shift
def test_color_shift_changes_appearance_but_not_geometry():
    sequence = synthetic.build("color_shift")
    first = sequence.frames[0][sequence.masks[0, 0]]
    last = sequence.frames[-1][sequence.masks[-1, 0]]

    assert not np.array_equal(first.mean(axis=0), last.mean(axis=0)), "colour should change"
    areas = sequence.masks[:, 0].reshape(sequence.n_frames, -1).sum(axis=1)
    assert len(set(areas.tolist())) == 1, "the object keeps its size"


# --------------------------------------------------------------------- prompts
@pytest.mark.parametrize("name", synthetic.available())
def test_prompts_land_inside_the_object(name):
    sequence = synthetic.build(name)
    for object_index in range(sequence.n_objects):
        prompt = sequence.prompt_for(object_index)
        point = prompt.points[0]
        assert prompt.frame_index == sequence.first_visible_frame(object_index)
        assert sequence.masks[prompt.frame_index, object_index][int(point.y), int(point.x)]
        assert point.positive is True


def test_prompts_are_reproducible():
    sequence = synthetic.build("occlusion")
    assert sequence.prompt_for(0) == sequence.prompt_for(0)


def test_asking_for_a_hidden_frame_raises():
    sequence = synthetic.build("occlusion")
    hidden = int(np.nonzero(~sequence.visible(0))[0][0])
    with pytest.raises(ValueError, match="not visible"):
        sequence.prompt_for(0, frame_index=hidden)


# ------------------------------------------------------------- interior_point
def test_interior_point_stays_inside_a_concave_shape():
    """A crescent's centroid is outside it, so a centroid prompt would be a lie."""
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    outer = (xx - 32) ** 2 + (yy - 32) ** 2 <= 24**2
    inner = (xx - 26) ** 2 + (yy - 30) ** 2 <= 20**2
    crescent = outer & ~inner

    point = interior_point(crescent)
    assert point is not None
    inside = crescent[int(point[1]), int(point[0])]
    assert inside, "the prompt must land on the object"

    centroid = np.array([xx[crescent].mean(), yy[crescent].mean()])
    assert not crescent[int(centroid[1]), int(centroid[0])], (
        "this fixture is only useful if the centroid really is outside"
    )


def test_interior_point_of_an_empty_mask_is_none():
    assert interior_point(np.zeros((8, 8), dtype=bool)) is None


# --------------------------------------------------------- sequence validation
def test_sequence_rejects_mismatched_frame_and_mask_counts():
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    masks = np.zeros((3, 1, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="disagree on T"):
        VideoSequence(name="bad", frames=frames, masks=masks)


def test_sequence_needs_a_frame_source():
    with pytest.raises(ValueError, match="one of frames / frame_dir"):
        VideoSequence(name="empty", masks=np.zeros((1, 1, 4, 4), dtype=bool))


def test_sequence_rejects_both_frame_sources(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        VideoSequence(
            name="both",
            frames=np.zeros((1, 4, 4, 3), dtype=np.uint8),
            masks=np.zeros((1, 1, 4, 4), dtype=bool),
            frame_dir=tmp_path,
        )


def test_sequence_frame_source_is_the_api_type():
    from app.models.frames import ArrayFrameSource

    sequence = synthetic.build("linear")
    source = sequence.frame_source()
    assert isinstance(source, ArrayFrameSource)
    assert len(source) == sequence.n_frames
    assert source.shape == sequence.shape
    assert np.array_equal(source[0], sequence.frames[0])


def test_fps_is_carried_into_the_frame_source():
    sequence = synthetic.build("linear", fps=7.5)
    assert sequence.frame_source().fps == pytest.approx(7.5)
