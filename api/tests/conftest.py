"""Shared pytest fixtures: import paths, an isolated workspace, synthetic video."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

API_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for path in (str(API_ROOT), str(TESTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.jobs import reset_job_manager  # noqa: E402
from app.pipeline import clear_preview_cache  # noqa: E402
from app.settings import reset_settings_cache  # noqa: E402
from app.storage import reset_workspace_cache  # noqa: E402
from app.video import ffmpeg_available  # noqa: E402


def _reset_caches() -> None:
    reset_settings_cache()
    reset_workspace_cache()
    reset_job_manager()
    clear_preview_cache()


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Point the API at a throwaway workspace so tests never touch real data."""
    monkeypatch.setenv("ROTOSTREAM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("ROTOSTREAM_LONG_SIDE", "192")
    monkeypatch.setenv("ROTOSTREAM_MAX_FRAMES", "64")
    _reset_caches()
    yield tmp_path / "workspace"
    _reset_caches()


requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"
)


def write_video(path: Path, frames: np.ndarray, fps: float = 10.0) -> Path:
    """Encode an (T, H, W, 3) uint8 array to H.264 mp4 via ffmpeg."""
    from PIL import Image

    tmp = path.parent / "_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame, mode="RGB").save(tmp / f"{index:06d}.png")

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps}",
            "-i", str(tmp / "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    for leftover in tmp.glob("*.png"):
        leftover.unlink()
    tmp.rmdir()
    return path


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """A 12-frame clip of a square moving across a plain background."""
    from synthetic import sliding_square

    sequence = sliding_square(n_frames=12, size=(96, 128))
    return write_video(tmp_path / "sample.mp4", sequence.frames, fps=10.0)
