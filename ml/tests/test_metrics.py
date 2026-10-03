"""Metric tests: hand-computed values, aggregation order, and reference parity."""
from __future__ import annotations

import numpy as np
import pytest

from rotostream_ml.metrics import (
    DEFAULT_BOUND_TH,
    DatasetScore,
    SequenceScore,
    boundary_f,
    boundary_tolerance,
    frame_jf,
    iou,
    j_and_f,
    score_dataset,
    score_sequence,
    seg2bmap,
)


# ------------------------------------------------------------------ region (J)
def test_iou_of_identical_masks_is_one():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    assert iou(mask, mask) == pytest.approx(1.0)


def test_iou_of_disjoint_masks_is_zero():
    left = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    right = np.zeros((8, 8), dtype=bool)
    right[:, 4:] = True
    assert iou(left, right) == pytest.approx(0.0)


def test_iou_of_two_empty_masks_is_one_by_convention():
    empty = np.zeros((4, 4), dtype=bool)
    assert iou(empty, empty) == pytest.approx(1.0)


def test_iou_hand_computed():
    # 8 True cells, 16 union -> 0.5
    prediction = np.zeros((4, 4), dtype=bool)
    prediction[0, :4] = True
    ground_truth = np.zeros((4, 4), dtype=bool)
    ground_truth[0, :2] = True
    ground_truth[1, :2] = True
    # intersection = 2, union = 4 + 4 - 2 = 6
    assert iou(prediction, ground_truth) == pytest.approx(2 / 6)


def test_iou_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        iou(np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool))


# ------------------------------------------------------------- boundary maps
def test_seg2bmap_of_an_interior_block_is_a_one_pixel_ring():
    """Pins the exact Martin et al. boundary, which is offset towards the origin.

    The block is deliberately interior. The reference algorithm overwrites the
    last row and column and then zeroes the corner, so a mask touching the image
    border legitimately loses those boundary pixels (a full-frame mask yields an
    empty boundary). That is an inherited quirk of the official implementation -
    faithfully reproduced here, and worth knowing before debugging a weird F.
    """
    seg = np.zeros((5, 5), dtype=bool)
    seg[1:4, 1:4] = True
    expected = np.array(
        [
            [True, True, True, True, False],
            [True, False, False, True, False],
            [True, False, False, True, False],
            [True, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    assert np.array_equal(seg2bmap(seg), expected)


def test_seg2bmap_of_a_full_frame_mask_is_empty_the_reference_quirk():
    """Documents the border behaviour above so nobody 'fixes' it by accident."""
    assert not seg2bmap(np.ones((3, 3), dtype=bool)).any()


def test_seg2bmap_of_an_empty_mask_is_empty():
    assert not seg2bmap(np.zeros((5, 5), dtype=bool)).any()


def test_seg2bmap_marks_the_inside_edge_for_a_hole():
    seg = np.ones((5, 5), dtype=bool)
    seg[2, 2] = False
    boundary = seg2bmap(seg)
    assert boundary[2, 1] or boundary[2, 2] or boundary[2, 3]
    assert boundary.sum() > 0


# ------------------------------------------------------------- boundary (F)
def test_boundary_f_of_identical_masks_is_one():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    assert boundary_f(mask, mask) == pytest.approx(1.0)


def test_boundary_f_of_distant_masks_is_zero():
    prediction = np.zeros((64, 64), dtype=bool)
    prediction[0:4, 0:4] = True
    ground_truth = np.zeros((64, 64), dtype=bool)
    ground_truth[60:64, 60:64] = True
    assert boundary_f(prediction, ground_truth) == pytest.approx(0.0)


def test_boundary_f_tolerates_a_one_pixel_shift():
    ground_truth = np.zeros((64, 64), dtype=bool)
    ground_truth[16:48, 16:48] = True
    prediction = np.zeros((64, 64), dtype=bool)
    prediction[17:49, 17:49] = True
    # Shifted by 1px, well inside the tolerance disk, so F stays at 1.
    assert boundary_f(prediction, ground_truth, bound_th=4) == pytest.approx(1.0)


def test_boundary_f_penalises_a_shift_beyond_tolerance():
    ground_truth = np.zeros((64, 64), dtype=bool)
    ground_truth[16:48, 16:48] = True
    prediction = np.zeros((64, 64), dtype=bool)
    prediction[26:58, 26:58] = True  # 10px off, tolerance is 1px here
    assert boundary_f(prediction, ground_truth, tolerance=1) < 0.5


def test_boundary_f_empty_prediction_against_a_real_object_is_zero():
    empty = np.zeros((32, 32), dtype=bool)
    ground_truth = np.zeros((32, 32), dtype=bool)
    ground_truth[8:24, 8:24] = True
    assert boundary_f(empty, ground_truth) == pytest.approx(0.0)


def test_boundary_f_hallucinated_object_against_empty_truth_is_zero():
    prediction = np.zeros((32, 32), dtype=bool)
    prediction[8:24, 8:24] = True
    empty = np.zeros((32, 32), dtype=bool)
    assert boundary_f(prediction, empty) == pytest.approx(0.0)


def test_boundary_f_of_two_empty_masks_is_one():
    empty = np.zeros((32, 32), dtype=bool)
    assert boundary_f(empty, empty) == pytest.approx(1.0)


# ------------------------------------------------------------------ tolerance
def test_boundary_tolerance_matches_the_official_formula_at_480p():
    # 0.008 * sqrt(480^2 + 854^2) = 7.84 -> 8px, the value DAVIS 480p uses.
    assert boundary_tolerance((480, 854)) == 8


def test_boundary_tolerance_is_one_pixel_for_a_64px_frame():
    assert boundary_tolerance((64, 64)) == 1


def test_boundary_tolerance_treats_large_values_as_absolute_pixels():
    assert boundary_tolerance((64, 64), bound_th=8) == 8


def test_boundary_tolerance_default_is_the_official_fraction():
    assert DEFAULT_BOUND_TH == 0.008


# ------------------------------------------------------------------ multi-object
def test_frame_jf_averages_over_objects():
    a_gt = np.zeros((16, 16), dtype=bool)
    a_gt[4:12, 4:12] = True
    b_gt = np.zeros((16, 16), dtype=bool)
    b_gt[0:4, 0:4] = True

    good = a_gt.copy()  # perfect
    bad = np.zeros((16, 16), dtype=bool)  # total miss
    j, f = frame_jf([good, bad], [a_gt, b_gt])

    assert j == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert f == pytest.approx(0.5)


def test_frame_jf_rejects_object_count_mismatch():
    gt = np.zeros((8, 8), dtype=bool)
    gt[2:6, 2:6] = True
    with pytest.raises(ValueError):
        frame_jf([gt], [gt, gt])


def test_frame_jf_rejects_a_frame_with_no_objects():
    with pytest.raises(ValueError):
        frame_jf([], [])


# ------------------------------------------------------------------ aggregation
def _block(size: int = 16, shifted: bool = False) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    offset = 2 if shifted else 0
    mask[4 + offset : 12 + offset, 4 + offset : 12 + offset] = True
    return mask


def test_score_sequence_averages_over_frames():
    gt = [_block() for _ in range(4)]
    pred = [_block(), _block(), _block(shifted=True), _block()]
    score = score_sequence("seq", pred, gt)

    assert score.n_frames == 4
    assert score.j == pytest.approx(np.mean([frame.j for frame in score.frames]))
    assert all(frame.n_objects == 1 for frame in score.frames)


def test_score_dataset_averages_over_sequences_not_frames():
    """A short perfect sequence must not be drowned out by a long bad one."""
    perfect = ([_block()] * 1, [_block()] * 1)
    bad = ([_block(shifted=True)] * 20, [_block()] * 20)

    good_score = score_sequence("perfect", *perfect)
    bad_score = score_sequence("bad", *bad)
    dataset = DatasetScore(sequences=[good_score, bad_score])

    # Sequence-mean: (1.0 + bad) / 2, which is clearly above frame-pooling.
    frame_pooled = (1.0 * 1 + bad_score.j * 20) / 21
    assert dataset.j == pytest.approx((1.0 + bad_score.j) / 2)
    assert dataset.j > frame_pooled


def test_score_dataset_builds_from_tuples():
    dataset = score_dataset(
        [
            ("a", [_block()], [_block()]),
            ("b", [_block()], [_block()]),
        ]
    )
    assert [s.name for s in dataset.sequences] == ["a", "b"]
    assert dataset.jf == pytest.approx(1.0)
    assert dataset.n_frames == 2


def test_score_sequence_rejects_a_frame_count_mismatch():
    with pytest.raises(ValueError):
        score_sequence("seq", [_block()], [_block(), _block()])


def test_two_objects_are_scored_separately_not_as_a_union():
    """Two disjoint objects score as two objects, so predicting their union hurts.

    This is why per-pixel mean IoU is not a substitute for the DAVIS metric: a
    union blob covers every object pixel, but it is not two objects.
    """
    a = np.zeros((16, 16), dtype=bool)
    a[2:6, 2:6] = True
    b = np.zeros((16, 16), dtype=bool)
    b[10:14, 10:14] = True
    empty = np.zeros((16, 16), dtype=bool)
    ground_truth = np.stack([a, b])

    perfect = score_sequence("two", [[a, b]], [ground_truth])
    assert perfect.frames[0].n_objects == 2
    assert perfect.j == pytest.approx(1.0)

    # One blob covering both, and nothing for the second object.
    collapsed = score_sequence("two", [[a | b, empty]], [ground_truth])
    assert collapsed.frames[0].n_objects == 2
    assert collapsed.j == pytest.approx(0.25)  # (0.5 for a-blob, 0.0 for the miss)
    assert collapsed.j < perfect.j


def test_scores_are_symmetric_between_prediction_and_truth_for_identical_masks():
    gt = _block()
    assert seg2bmap(gt).sum() == seg2bmap(gt.copy()).sum()
    j, f = j_and_f(gt, gt)
    assert (j, f) == (pytest.approx(1.0), pytest.approx(1.0))


def test_sequence_score_of_no_frames_is_zero():
    assert SequenceScore(name="empty").jf == 0.0
    assert DatasetScore().jf == 0.0


# ------------------------------------------------------------------ reference parity
def _reference_boundary_f(fg: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """The official implementation verbatim, using scikit-image."""
    from skimage.morphology import binary_dilation, disk

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(fg.shape))
    fg_boundary = seg2bmap(fg)
    gt_boundary = seg2bmap(gt)
    fg_dil = binary_dilation(fg_boundary, footprint=disk(int(bound_pix)))
    gt_dil = binary_dilation(gt_boundary, footprint=disk(int(bound_pix)))
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


@pytest.mark.parametrize("bound_th", [0.008, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_skimage_reference(seed, bound_th):
    """Our scipy dilation must equal the official scikit-image dilation path.

    skimage is not a harness dependency, so this is skipped when it is absent -
    but while it is installed it is the cheapest way to know the F numbers are
    comparable to the published ones.
    """
    pytest.importorskip("skimage")
    rng = np.random.default_rng(seed)
    size = 96
    gt = rng.random((size, size)) > 0.75
    pred = rng.random((size, size)) > 0.75
    assert sum(map(int, gt.flatten())) > 0 and pred.any()

    ours = boundary_f(pred, gt, bound_th)
    theirs = _reference_boundary_f(pred, gt, bound_th)
    assert ours == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("radius", [0, 1, 3, 8])
@pytest.mark.parametrize("seed", [13, 28])
def test_distance_dilation_matches_disk_reference_on_sparse_boundaries(radius, seed):
    """Sparse edge-touching shapes exercise disk distances and image borders."""
    pytest.importorskip("skimage")
    rng = np.random.default_rng(seed)
    gt = np.zeros((96, 112), dtype=bool)
    gt[0:48, :20] = True
    gt[70:90, 105:] = True
    pred = np.roll(gt, shift=int(rng.integers(2, 9)), axis=1)
    assert boundary_f(pred, gt, tolerance=radius) == pytest.approx(
        _reference_boundary_f(pred, gt, bound_th=radius if radius else 0), abs=1e-12)
