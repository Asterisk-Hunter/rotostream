"""Mask IO, overlay rendering and run-length encoding.

Masks are stored on disk as 8-bit PNGs (0 or 255) at the **working frame
resolution**. Overlays are rendered server-side as RGBA PNGs so the browser can
simply stack them on the frame image with no client-side pixel work.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

#: Outline + fill colour for the studio overlay (sky-400).
ACCENT: tuple[int, int, int] = (56, 189, 248)


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def load_mask_or_empty(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Missing masks are treated as all-False so partial sessions still export."""
    try:
        mask = load_mask(path)
    except (FileNotFoundError, OSError):
        return np.zeros(shape, dtype=bool)
    return _fit(mask, shape)


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == tuple(shape):
        return mask
    height, width = shape
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized > 0


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, optimize=True)
    return path


def mask_png_bytes(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def overlay_rgba(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = ACCENT,
    alpha: float = 0.38,
    outline: int = 2,
) -> np.ndarray:
    """Tint the mask and draw a crisp outline, as an ``(H, W, 4)`` RGBA array."""
    height, width = frame.shape[:2]
    mask = _fit(np.asarray(mask, dtype=bool), (height, width))

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = np.where(mask, int(alpha * 255), 0).astype(np.uint8)

    if outline > 0 and mask.any():
        kernel = np.ones((outline * 2 + 1, outline * 2 + 1), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
        edge = mask & ~eroded
        rgba[edge, 3] = 235
        rgba[edge, 0], rgba[edge, 1], rgba[edge, 2] = color

    return rgba


def overlay_png_bytes(frame: np.ndarray, mask: np.ndarray, **kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(overlay_rgba(frame, mask, **kwargs), mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def outline_mask(mask: np.ndarray, width: int = 2) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return mask & ~eroded


# ----------------------------------------------------------------------- RLE
def rle_encode(mask: np.ndarray) -> list[int]:
    """COCO-style run-length encoding (column-major, leading zero-run included)."""
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    if pixels.size == 0:
        return []
    boundaries = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [pixels.size]))
    counts = (ends - starts).astype(int).tolist()
    if pixels[0] != 0:
        counts.insert(0, 0)
    return counts


def rle_decode(counts: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`rle_encode`. Used by the test-suite to verify round-trips."""
    height, width = shape
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    value = 0
    for run in counts:
        if position >= flat.size:
            break
        end = min(position + int(run), flat.size)
        if value:
            flat[position:end] = 1
        position = end
        value = 1 - value
    return flat.reshape((height, width), order="F").astype(bool)
