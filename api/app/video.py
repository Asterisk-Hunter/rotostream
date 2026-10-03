"""ffmpeg / ffprobe integration: probe, extract frames, composite and encode exports."""
from __future__ import annotations

import json
import io
import math
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from . import masks as mask_utils
from .settings import get_settings

Progress = Callable[[float, str], None]


class FFmpegError(RuntimeError):
    """ffmpeg or ffprobe failed, or is not installed."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: Sequence[Any], *, cwd: Path | None = None) -> None:
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=get_settings().ffmpeg_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("media processing exceeded the configured time limit") from exc
    except FileNotFoundError as exc:  # pragma: no cover - depends on host
        raise FFmpegError(f"ffmpeg/ffprobe not found on PATH: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        raise FFmpegError(f"{Path(str(cmd[0])).name} failed ({proc.returncode}):\n{tail}")


# --------------------------------------------------------------------- probe
@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration_s: float
    n_frames: int
    has_audio: bool


def _ratio(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            denominator_f = float(denominator)
            return float(numerator) / denominator_f if denominator_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> VideoProbe:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    try:
        proc = subprocess.run([str(part) for part in cmd], capture_output=True, text=True,
                              timeout=min(30, get_settings().ffmpeg_timeout_s))
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("video probing exceeded the configured time limit") from exc
    except FileNotFoundError as exc:
        raise FFmpegError("ffprobe not found on PATH") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-4:])
        raise FFmpegError(f"ffprobe failed ({proc.returncode}):\n{tail}")

    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError("no video stream found in the uploaded file")

    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate")) or 30.0
    if not math.isfinite(fps) or fps <= 0 or fps > 1000:
        raise FFmpegError("video has an invalid frame rate")
    duration = 0.0
    for source in (video.get("duration"), payload.get("format", {}).get("duration")):
        try:
            duration = float(source)
            if not math.isfinite(duration):
                duration = 0.0
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    n_frames = 0
    try:
        n_frames = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        n_frames = 0
    if n_frames <= 0 and duration > 0 and fps > 0:
        n_frames = int(round(duration * fps))

    return VideoProbe(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        duration_s=duration,
        n_frames=n_frames,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


# ------------------------------------------------------------ frame extraction
def target_frame_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Scale down so the long side is at most ``long_side``, keeping even dimensions."""
    if width <= 0 or height <= 0:
        return long_side, long_side
    scale = min(1.0, long_side / float(max(width, height)))
    new_w = max(2, int(round(width * scale)))
    new_h = max(2, int(round(height * scale)))
    return new_w - (new_w % 2), new_h - (new_h % 2)


def _jpeg_qscale(quality: int) -> str:
    """Map 0-100 quality onto ffmpeg's 2 (best) .. 31 (worst) qscale."""
    quality = max(1, min(100, int(quality)))
    return str(max(2, min(31, round(31 - 29 * quality / 100))))


def extract_frames(
    source: str | Path,
    out_dir: str | Path,
    *,
    long_side: int = 960,
    quality: int = 92,
    max_frames: int = 900,
) -> tuple[int, int, int]:
    """Decode to ``out_dir/000000.jpg`` ... Returns ``(n_frames, width, height)``."""
    info = probe(source)
    if info.width <= 0 or info.height <= 0:
        raise FFmpegError("video has invalid dimensions")
    if info.width * info.height > get_settings().max_source_pixels:
        raise FFmpegError("source resolution exceeds ROTOSTREAM_MAX_SOURCE_PIXELS")
    width, height = target_frame_size(info.width, info.height, long_side)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", source,
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-q:v", _jpeg_qscale(quality),
            "-frames:v", int(max_frames),
            "-start_number", "0",
            "-fps_mode", "passthrough",
            out_dir / "%06d.jpg",
        ]
    )
    written = sorted(out_dir.glob("*.jpg"))
    if not written:
        raise FFmpegError("frame extraction produced no output")
    return len(written), width, height


def read_frame(frames_dir: str | Path, index: int) -> np.ndarray:
    """Decode one extracted frame as ``(H, W, 3)`` uint8 RGB."""
    path = Path(frames_dir) / f"{index:06d}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"frame {index} not extracted")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


# ------------------------------------------------------------------- exports
@dataclass
class ExportInputs:
    frames_dir: Path
    masks_dir: Path
    n_frames: int
    fps: float
    width: int
    height: int
    out_path: Path

    def frame(self, index: int) -> np.ndarray:
        path = self.frames_dir / f"{index:06d}.jpg"
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def mask(self, index: int) -> np.ndarray:
        return mask_utils.load_mask_or_empty(
            self.masks_dir / f"{index:06d}.png", (self.height, self.width)
        )


def _encode_png_sequence(
    tmp_dir: Path, fps: float, out_path: Path, encoder_args: Sequence[Any]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-framerate", f"{max(fps, 1.0):.6f}",
            "-i", tmp_dir / "%06d.png",
            *encoder_args,
            "-movflags", "+faststart",
            out_path,
        ]
    )


def _tint(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = frame.astype(np.float32)
    if mask.any():
        colour = np.array(mask_utils.ACCENT, dtype=np.float32)
        out[mask] = out[mask] * (1 - alpha) + colour * alpha
        edge = mask_utils.outline_mask(mask, width=2)
        out[edge] = colour
    return np.clip(out, 0, 255).astype(np.uint8)


def _background(frame: np.ndarray, mode: str, blur_radius: int) -> np.ndarray:
    if mode == "black":
        return np.zeros_like(frame)
    if mode == "white":
        return np.full_like(frame, 255)
    if mode == "green":
        out = np.zeros_like(frame)
        out[..., 0] = 0
        out[..., 1] = 177
        out[..., 2] = 64
        return out
    radius = max(1, blur_radius) | 1
    return cv2.GaussianBlur(frame, (radius, radius), 0)


def export_alpha_webm(inp: ExportInputs, progress: Progress) -> Path:
    tmp = inp.out_path.parent / f".tmp_{inp.out_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(inp.n_frames):
            frame = inp.frame(index)
            mask = inp.mask(index)
            rgba = np.dstack([frame, (mask * 255).astype(np.uint8)])
            Image.fromarray(rgba, mode="RGBA").save(tmp / f"{index:06d}.png")
            progress(0.85 * (index + 1) / inp.n_frames, f"compositing {index + 1}/{inp.n_frames}")
        progress(0.88, "encoding VP9 with alpha")
        _encode_png_sequence(
            tmp, inp.fps, inp.out_path,
            ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "32",
             "-auto-alt-ref", "0", "-row-mt", "1", "-cpu-used", "4"],
        )
        progress(1.0, f"wrote {inp.out_path.name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return inp.out_path


def export_overlay_mp4(inp: ExportInputs, progress: Progress) -> Path:
    tmp = inp.out_path.parent / f".tmp_{inp.out_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(inp.n_frames):
            Image.fromarray(_tint(inp.frame(index), inp.mask(index)), mode="RGB").save(
                tmp / f"{index:06d}.png"
            )
            progress(0.85 * (index + 1) / inp.n_frames, f"compositing {index + 1}/{inp.n_frames}")
        progress(0.88, "encoding H.264")
        _encode_png_sequence(
            tmp, inp.fps, inp.out_path,
            ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"],
        )
        progress(1.0, f"wrote {inp.out_path.name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return inp.out_path


def export_replace_bg(inp: ExportInputs, progress: Progress, *, background: str = "blur",
                      blur_radius: int = 24) -> Path:
    tmp = inp.out_path.parent / f".tmp_{inp.out_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(inp.n_frames):
            frame = inp.frame(index)
            mask = inp.mask(index)
            back = _background(frame, background, blur_radius)
            composite = np.where(mask[..., None], frame, back)
            Image.fromarray(composite.astype(np.uint8), mode="RGB").save(tmp / f"{index:06d}.png")
            progress(0.85 * (index + 1) / inp.n_frames, f"compositing {index + 1}/{inp.n_frames}")
        progress(0.88, "encoding H.264")
        _encode_png_sequence(
            tmp, inp.fps, inp.out_path,
            ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"],
        )
        progress(1.0, f"wrote {inp.out_path.name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return inp.out_path


def export_cutout_zip(inp: ExportInputs, progress: Progress) -> Path:
    with zipfile.ZipFile(inp.out_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(inp.n_frames):
            frame = inp.frame(index)
            mask = inp.mask(index)
            rgba = np.dstack([frame, (mask * 255).astype(np.uint8)])
            buffer = io.BytesIO()
            Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
            archive.writestr(f"cutout/{index:06d}.png", buffer.getvalue())
            progress(0.95 * (index + 1) / inp.n_frames, f"packing {index + 1}/{inp.n_frames}")
    progress(1.0, f"wrote {inp.out_path.name}")
    return inp.out_path


def export_mask_zip(inp: ExportInputs, progress: Progress) -> Path:
    with zipfile.ZipFile(inp.out_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(inp.n_frames):
            path = inp.masks_dir / f"{index:06d}.png"
            if path.exists():
                archive.write(path, arcname=f"masks/{index:06d}.png")
            progress(0.95 * (index + 1) / inp.n_frames, f"packing {index + 1}/{inp.n_frames}")
    progress(1.0, f"wrote {inp.out_path.name}")
    return inp.out_path


def export_mask_rle(inp: ExportInputs, progress: Progress) -> Path:
    records: list[dict[str, Any]] = []
    for index in range(inp.n_frames):
        mask = inp.mask(index)
        records.append(
            {
                "frame_index": index,
                "size": [inp.height, inp.width],
                "counts": mask_utils.rle_encode(mask),
                "area": int(mask.sum()),
            }
        )
        progress(0.95 * (index + 1) / inp.n_frames, f"encoding {index + 1}/{inp.n_frames}")
    payload = {
        "format": "coco_rle",
        "fps": inp.fps,
        "height": inp.height,
        "width": inp.width,
        "n_frames": inp.n_frames,
        "frames": records,
    }
    inp.out_path.write_text(json.dumps(payload), encoding="utf-8")
    progress(1.0, f"wrote {inp.out_path.name}")
    return inp.out_path


EXPORTERS: dict[str, Callable[..., Path]] = {
    "alpha_webm": export_alpha_webm,
    "overlay_mp4": export_overlay_mp4,
    "cutout_zip": export_cutout_zip,
    "mask_zip": export_mask_zip,
    "mask_rle_json": export_mask_rle,
    "replace_bg": export_replace_bg,
}

EXPORT_SUFFIX: dict[str, str] = {
    "alpha_webm": ".webm",
    "overlay_mp4": ".mp4",
    "cutout_zip": ".zip",
    "mask_zip": ".zip",
    "mask_rle_json": ".json",
    "replace_bg": ".mp4",
}


def run_export(kind: str, inp: ExportInputs, progress: Progress, **options: Any) -> Path:
    if kind not in EXPORTERS:
        raise ValueError(f"unknown export kind {kind!r}; known: {', '.join(sorted(EXPORTERS))}")
    if inp.n_frames <= 0 or inp.width <= 0 or inp.height <= 0:
        raise ValueError("export requires extracted video frames")
    inp.out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return EXPORTERS[kind](inp, progress, **options)
    except BaseException:
        inp.out_path.unlink(missing_ok=True)
        raise
