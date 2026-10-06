"""ffmpeg helpers, mask IO/RLE and every export kind."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from app import masks as mask_utils
from app.video import (
    EXPORT_SUFFIX,
    ExportInputs,
    extract_frames,
    ffmpeg_available,
    probe,
    read_frame,
    run_export,
    target_frame_size,
)
from marks import requires_ffmpeg


# ------------------------------------------------------------ geometry helpers
def test_target_frame_size_preserves_aspect_and_even_dimensions():
    assert target_frame_size(1920, 1080, 960) == (960, 540)
    assert target_frame_size(1080, 1920, 960) == (540, 960)
    assert target_frame_size(320, 240, 960) == (320, 240), "must not upscale"
    for width, height in [(1921, 1081), (999, 333), (7, 5)]:
        new_w, new_h = target_frame_size(width, height, 720)
        assert new_w % 2 == 0 and new_h % 2 == 0
        assert new_w >= 2 and new_h >= 2


def test_target_frame_size_handles_degenerate_input():
    assert target_frame_size(0, 0, 512) == (512, 512)


# ------------------------------------------------------------------ mask utils
def test_rle_round_trip_is_lossless():
    rng = np.random.default_rng(0)
    mask = rng.random((17, 23)) > 0.6
    counts = mask_utils.rle_encode(mask)
    assert mask_utils.rle_decode(counts, mask.shape).tolist() == mask.tolist()


def test_rle_of_empty_and_full_masks():
    empty = np.zeros((4, 5), bool)
    full = np.ones((4, 5), bool)
    assert mask_utils.rle_encode(empty) == [20]
    assert mask_utils.rle_encode(full) == [0, 20]
    assert not mask_utils.rle_decode(mask_utils.rle_encode(empty), empty.shape).any()
    assert mask_utils.rle_decode(mask_utils.rle_encode(full), full.shape).all()


def test_mask_save_and_load_round_trip(tmp_path):
    mask = np.zeros((12, 20), bool)
    mask[3:9, 4:16] = True
    path = mask_utils.save_mask(tmp_path / "nested" / "000000.png", mask)
    assert path.exists()
    assert np.array_equal(mask_utils.load_mask(path), mask)


def test_missing_mask_loads_as_empty():
    mask = mask_utils.load_mask_or_empty("/nonexistent/000000.png", (8, 6))
    assert mask.shape == (8, 6) and not mask.any()


def test_overlay_is_rgba_with_an_opaque_outline():
    frame = np.zeros((32, 32, 3), np.uint8)
    mask = np.zeros((32, 32), bool)
    mask[10:20, 10:20] = True
    rgba = mask_utils.overlay_rgba(frame, mask)
    assert rgba.shape == (32, 32, 4)
    assert rgba[..., 3].max() >= 200, "the outline should be near-opaque"
    assert rgba[10, 10, 3] >= 200, "the boundary pixel is part of the outline"


def test_overlay_accepts_a_mismatched_mask_resolution():
    frame = np.zeros((40, 60, 3), np.uint8)
    rgba = mask_utils.overlay_rgba(frame, np.ones((10, 15), bool))
    assert rgba.shape == (40, 60, 4)
    assert (rgba[..., 3] > 0).any()


def test_outline_of_an_empty_mask_is_empty():
    assert not mask_utils.outline_mask(np.zeros((8, 8), bool)).any()


# --------------------------------------------------------------- extraction
@requires_ffmpeg
def test_probe_reports_geometry(tmp_path, sample_video):
    info = probe(sample_video)
    assert (info.width, info.height) == (128, 96)
    assert info.fps == pytest.approx(10.0, rel=0.05)
    assert info.n_frames == 12
    assert info.has_audio is False


@requires_ffmpeg
def test_extract_frames_writes_readable_jpegs(tmp_path, sample_video):
    count, width, height = extract_frames(
        sample_video, tmp_path / "frames", long_side=192, quality=92, max_frames=64
    )
    assert (count, width, height) == (12, 128, 96)
    frame = read_frame(tmp_path / "frames", 0)
    assert frame.shape == (96, 128, 3) and frame.dtype == np.uint8
    assert frame.std() > 0, "decoded frame should not be a flat image"


@requires_ffmpeg
def test_extraction_respects_the_frame_budget(tmp_path, sample_video):
    count, _, _ = extract_frames(
        sample_video, tmp_path / "frames", long_side=64, quality=80, max_frames=5
    )
    assert count == 5


@requires_ffmpeg
def test_read_frame_raises_for_a_missing_index(tmp_path, sample_video):
    extract_frames(sample_video, tmp_path / "frames", long_side=64, quality=80, max_frames=3)
    with pytest.raises(FileNotFoundError):
        read_frame(tmp_path / "frames", 99)


# ------------------------------------------------------------------ exports
def _export_inputs(tmp_path, sample_video):
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    count, width, height = extract_frames(
        sample_video, frames_dir, long_side=128, quality=80, max_frames=6
    )
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        mask = np.zeros((height, width), bool)
        mask[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = True
        mask_utils.save_mask(masks_dir / f"{index:06d}.png", mask)
    return count, width, height, frames_dir, masks_dir


@requires_ffmpeg
@pytest.mark.parametrize(
    "kind",
    ["mask_zip", "mask_rle_json", "overlay_mp4", "alpha_webm", "cutout_zip", "replace_bg"],
)
def test_every_export_kind_produces_an_artifact(tmp_path, sample_video, kind):
    count, width, height, frames_dir, masks_dir = _export_inputs(tmp_path, sample_video)
    out_path = tmp_path / f"out{EXPORT_SUFFIX[kind]}"
    inputs = ExportInputs(
        frames_dir=frames_dir, masks_dir=masks_dir, n_frames=count,
        fps=10.0, width=width, height=height, out_path=out_path,
    )
    progress: list[float] = []
    run_export(kind, inputs, lambda fraction, message="": progress.append(fraction))

    assert out_path.exists() and out_path.stat().st_size > 0
    assert progress and max(progress) >= 0.9, f"progress never advanced: {progress}"
    assert not (tmp_path / f".tmp_{out_path.stem}").exists(), "temp dir should be cleaned up"


@requires_ffmpeg
def test_streamed_mp4_export_keeps_source_audio(tmp_path, sample_video):
    count, width, height, frames_dir, masks_dir = _export_inputs(tmp_path, sample_video)
    audio_source = tmp_path / "with-audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", str(sample_video),
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100:duration=1.2",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(audio_source),
        ],
        check=True,
        capture_output=True,
    )
    out_path = tmp_path / "with-audio-export.mp4"
    inputs = ExportInputs(
        frames_dir=frames_dir, masks_dir=masks_dir, n_frames=count,
        fps=10.0, width=width, height=height, out_path=out_path,
        audio_source=audio_source,
    )
    run_export("overlay_mp4", inputs, lambda fraction, message="": None)
    assert probe(out_path).has_audio


@requires_ffmpeg
def test_export_skips_frames_with_no_mask(tmp_path, sample_video):
    """A cancelled session leaves gaps; exports must still complete."""
    count, width, height, frames_dir, masks_dir = _export_inputs(tmp_path, sample_video)
    (masks_dir / "000002.png").unlink()

    out_path = tmp_path / "out.zip"
    inputs = ExportInputs(
        frames_dir=frames_dir, masks_dir=masks_dir, n_frames=count,
        fps=10.0, width=width, height=height, out_path=out_path,
    )
    run_export("mask_zip", inputs, lambda fraction, message="": None)
    assert out_path.stat().st_size > 0


@requires_ffmpeg
def test_rle_export_contains_one_record_per_frame(tmp_path, sample_video):
    import json

    count, width, height, frames_dir, masks_dir = _export_inputs(tmp_path, sample_video)
    out_path = tmp_path / "masks.json"
    inputs = ExportInputs(
        frames_dir=frames_dir, masks_dir=masks_dir, n_frames=count,
        fps=10.0, width=width, height=height, out_path=out_path,
    )
    run_export("mask_rle_json", inputs, lambda fraction, message="": None)

    payload = json.loads(out_path.read_text())
    assert payload["format"] == "coco_rle"
    assert payload["n_frames"] == count
    assert len(payload["frames"]) == count
    assert payload["frames"][0]["size"] == [height, width]
    assert payload["frames"][0]["area"] > 0


@requires_ffmpeg
def test_unknown_export_kind_is_rejected(tmp_path, sample_video):
    count, width, height, frames_dir, masks_dir = _export_inputs(tmp_path, sample_video)
    inputs = ExportInputs(
        frames_dir=frames_dir, masks_dir=masks_dir, n_frames=count,
        fps=10.0, width=width, height=height, out_path=tmp_path / "x",
    )
    with pytest.raises(ValueError, match="unknown export kind"):
        run_export("gif_of_my_cat", inputs, lambda fraction, message="": None)
