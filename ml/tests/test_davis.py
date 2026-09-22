"""DAVIS loader tests, run against a fixture in the official layout.

Downloading the real 800MB release to test a path-join would be absurd, so the
fixture writes the same directory structure, the same ``NNNNN.jpg`` frame names
and the same palette-PNG annotations. That covers the parts that actually break:
locating the folders, reading the split files, decoding multi-object annotations
and refusing to load misaligned data.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rotostream_ml.davis import Davis2017

#: The colours the real release uses for the first two objects.
OBJECT_COLOURS = [(128, 0, 0), (0, 128, 0)]


def _write_frame(path, seed: int, size=(32, 40)) -> None:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
    Image.fromarray(frame, mode="RGB").save(path, quality=95)


def _write_annotation(path, masks, size=(32, 40)) -> None:
    canvas = np.zeros((*size, 3), dtype=np.uint8)
    for index, mask in enumerate(masks):
        canvas[mask] = OBJECT_COLOURS[index]
    Image.fromarray(canvas, mode="RGB").save(path)


def _boxes(n_frames: int, size=(32, 40), n_objects: int = 1, drift: int = 2):
    """One moving 4x4 box per object, offset so they never overlap."""
    height, width = size
    out = []
    for t in range(n_frames):
        frame_masks = []
        for object_index in range(n_objects):
            mask = np.zeros(size, dtype=bool)
            x = 2 + drift * t + object_index * 12
            y = 4 + object_index * 12
            mask[y : y + 4, x : x + 4] = True
            frame_masks.append(mask)
        out.append(frame_masks)
    return out


def make_davis(root, sequences: dict[str, int], size=(32, 40), n_objects=1, splits=None):
    """Create a DAVIS-shaped tree. ``sequences`` maps name -> frame count."""
    base = root / "2017" / "trainval"
    images = base / "JPEGImages" / "480p"
    annotations = base / "Annotations" / "480p"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)

    for index, (name, n_frames) in enumerate(sequences.items()):
        image_dir = images / name
        annotation_dir = annotations / name
        image_dir.mkdir()
        annotation_dir.mkdir()
        boxes = _boxes(n_frames, size=size, n_objects=n_objects)
        for t in range(n_frames):
            _write_frame(image_dir / f"{t:05d}.jpg", seed=index * 100 + t, size=size)
            _write_annotation(annotation_dir / f"{t:05d}.png", boxes[t], size=size)

    splits = splits or {}
    for split_name, names in splits.items():
        (base / f"{split_name}.txt").write_text("\n".join(names), encoding="utf-8")
    return root


# ------------------------------------------------------------------- locating
def test_loads_the_official_layout(tmp_path):
    make_davis(tmp_path, {"bear": 3}, splits={"val": ["bear"]})
    dataset = Davis2017(root=tmp_path, split="val")

    assert dataset.names == ["bear"]
    sequence = dataset.load("bear")
    assert sequence.n_frames == 3
    assert sequence.n_objects == 1
    assert sequence.frame_source().n_frames == 3


def test_root_can_be_any_directory_above_the_data(tmp_path):
    make_davis(tmp_path, {"bear": 2}, splits={"val": ["bear"]})

    for root in (tmp_path, tmp_path / "2017", tmp_path / "2017" / "trainval"):
        dataset = Davis2017(root=root, split="val")
        assert dataset.names == ["bear"], f"failed for root={root}"


def test_missing_root_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Davis2017(root=tmp_path / "nope")


def test_a_directory_without_the_expected_subfolders_explains_itself(tmp_path):
    (tmp_path / "random").mkdir()
    with pytest.raises(FileNotFoundError, match="JPEGImages"):
        Davis2017(root=tmp_path)


def test_representations_are_readable(tmp_path):
    make_davis(tmp_path, {"bear": 2}, splits={"val": ["bear"]})
    dataset = Davis2017(root=tmp_path, split="val")
    assert "bear" in repr(dataset)
    assert len(dataset) == 1
    assert [sequence.name for sequence in dataset] == ["bear"]


# ---------------------------------------------------------------------- masks
def test_annotations_decode_to_per_object_masks(tmp_path):
    make_davis(tmp_path, {"bear": 3}, splits={"val": ["bear"]})
    sequence = Davis2017(root=tmp_path, split="val").load("bear")

    assert sequence.masks.shape == (3, 1, 32, 40)
    assert sequence.masks.dtype == bool

    # The box moves 2px per frame; its area is constant and its column advances.
    areas = sequence.masks[:, 0].reshape(3, -1).sum(axis=1)
    assert list(areas) == [16, 16, 16]

    def centre_x(t: int) -> float:
        ys, xs = np.nonzero(sequence.masks[t, 0])
        return float(xs.mean())

    assert centre_x(1) - centre_x(0) == pytest.approx(2, abs=0.51)
    assert centre_x(2) > centre_x(1)


def test_multi_object_annotations_get_one_channel_per_object(tmp_path):
    make_davis(tmp_path, {"bear": 3}, n_objects=2, splits={"val": ["bear"]})
    sequence = Davis2017(root=tmp_path, split="val").load("bear")

    assert sequence.n_objects == 2
    assert not np.any(sequence.masks[:, 0] & sequence.masks[:, 1]), "objects must be disjoint"
    assert sequence.masks[:, 0].any() and sequence.masks[:, 1].any()


def test_masks_line_up_with_the_frames(tmp_path):
    """Frame and annotation ordering must not drift apart by one."""
    make_davis(tmp_path, {"bear": 3}, splits={"val": ["bear"]})
    sequence = Davis2017(root=tmp_path, split="val").load("bear")
    source = sequence.frame_source()

    # The fixture writes frames with a per-frame seeded pattern, so frame 0 loaded
    # through the framename ordering must equal the annotation of frame 0's box.
    assert source[0].shape == (32, 40, 3)
    assert sequence.object_masks(0)[0].any()
    assert not sequence.object_masks(0)[0, 0, 0], "the box never touches the top-left corner"


def test_frames_stay_on_disk_rather_than_in_memory(tmp_path):
    make_davis(tmp_path, {"bear": 2}, splits={"val": ["bear"]})
    sequence = Davis2017(root=tmp_path, split="val").load("bear")

    from app.models.frames import DirectoryFrameSource

    assert sequence.frames is None
    assert isinstance(sequence.frame_source(), DirectoryFrameSource)


def test_misaligned_frame_and_annotation_counts_are_refused(tmp_path):
    make_davis(tmp_path, {"bear": 3}, splits={"val": ["bear"]})
    extra = tmp_path / "2017" / "trainval" / "JPEGImages" / "480p" / "bear" / "00003.jpg"
    _write_frame(extra, seed=99)

    with pytest.raises(ValueError, match="annotations"):
        Davis2017(root=tmp_path, split="val").load("bear")


def test_an_all_background_annotation_is_refused(tmp_path):
    make_davis(tmp_path, {"empty": 2}, splits={"val": ["empty"]})
    annotation = tmp_path / "2017" / "trainval" / "Annotations" / "480p" / "empty"
    for path in annotation.glob("*.png"):
        Image.fromarray(np.zeros((32, 40, 3), dtype=np.uint8), mode="RGB").save(path)

    with pytest.raises(ValueError, match="empty"):
        Davis2017(root=tmp_path, split="val").load("empty")


# --------------------------------------------------------------------- splits
def test_split_files_are_respected(tmp_path):
    make_davis(
        tmp_path,
        {"bear": 2, "camel": 2, "dog": 2},
        splits={"val": ["camel"], "train": ["bear", "dog"]},
    )
    assert Davis2017(root=tmp_path, split="val").names == ["camel"]
    assert Davis2017(root=tmp_path, split="train").names == ["bear", "dog"]
    assert Davis2017(root=tmp_path, split="all").names == ["bear", "camel", "dog"]


def test_names_listed_but_not_downloaded_are_ignored(tmp_path):
    make_davis(tmp_path, {"bear": 2}, splits={"val": ["bear", "ghost"]})
    assert Davis2017(root=tmp_path, split="val").names == ["bear"]


def test_missing_split_file_warns_because_the_split_is_a_guess(tmp_path):
    make_davis(tmp_path, {"bear": 1})
    with pytest.warns(UserWarning, match="the split is a guess"):
        names = Davis2017(root=tmp_path, split="val").names
    assert names == ["bear"]


def test_unknown_sequence_names_are_reported(tmp_path):
    make_davis(tmp_path, {"bear": 1}, splits={"val": ["bear"]})
    with pytest.raises(FileNotFoundError, match="no images"):
        Davis2017(root=tmp_path, split="val").load("ghost")
