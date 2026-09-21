"""Minimal, dependency-free RGB frame containers used by the tracker contract.

A ``FrameSource`` is random-access and lazy: implementations must not require the
whole video to be materialised in RAM. Trackers receive one of these from
``set_video`` and may index it freely while propagating.
"""
from __future__ import annotations

import abc
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


class FrameSource(abc.ABC):
    """Random-access RGB frames, as ``uint8`` arrays of shape ``(H, W, 3)``."""

    @property
    @abc.abstractmethod
    def n_frames(self) -> int: ...

    @property
    @abc.abstractmethod
    def width(self) -> int: ...

    @property
    @abc.abstractmethod
    def height(self) -> int: ...

    @property
    @abc.abstractmethod
    def fps(self) -> float: ...

    @abc.abstractmethod
    def __getitem__(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``(H, W, 3) uint8 RGB``.

        Negative indices follow normal Python semantics. Out-of-range indices
        must raise ``IndexError``.
        """

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def indices(self) -> range:
        return range(self.n_frames)

    def __iter__(self):
        for i in self.indices():
            yield self[i]


class ArrayFrameSource(FrameSource):
    """In-memory frames. Used for tests, synthetic sequences and small clips."""

    def __init__(self, frames: np.ndarray, fps: float = 30.0):
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be (T, H, W, 3), got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        self._frames = frames
        self._fps = float(fps)

    @property
    def n_frames(self) -> int:
        return int(self._frames.shape[0])

    @property
    def height(self) -> int:
        return int(self._frames.shape[1])

    @property
    def width(self) -> int:
        return int(self._frames.shape[2])

    @property
    def fps(self) -> float:
        return self._fps

    def __getitem__(self, index: int) -> np.ndarray:
        return self._frames[index]


class DirectoryFrameSource(FrameSource):
    """Frames previously extracted to disk as ``000000.jpg``, ``000001.jpg``, ...

    Keeps a small LRU of decoded frames: trackers typically re-read a handful of
    recent frames, and JPEG decode dominates otherwise.
    """

    def __init__(self, directory: str | Path, fps: float = 30.0, cache_size: int = 8):
        self.directory = Path(directory)
        self._paths = sorted(self.directory.glob("*.jpg")) + sorted(self.directory.glob("*.png"))
        self._paths = sorted(self._paths)
        if not self._paths:
            raise FileNotFoundError(f"no frames found in {self.directory}")
        self._fps = float(fps)
        self._cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        with Image.open(self._paths[0]) as first:
            self._width, self._height = first.size

    @property
    def n_frames(self) -> int:
        return len(self._paths)

    @property
    def width(self) -> int:
        return int(self._width)

    @property
    def height(self) -> int:
        return int(self._height)

    @property
    def fps(self) -> float:
        return self._fps

    def path_for(self, index: int) -> Path:
        return self._paths[index]

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index += len(self._paths)
        if not 0 <= index < len(self._paths):
            raise IndexError(index)

        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        with Image.open(self._paths[index]) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)

        if self._cache_size:
            self._cache[index] = frame
            self._cache.move_to_end(index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return frame
