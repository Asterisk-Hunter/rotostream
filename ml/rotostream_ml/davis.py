"""DAVIS 2017 loader.

Reads the standard release layout::

    DAVIS/
      2017/
        trainval/
          JPEGImages/480p/<sequence>/00000.jpg
          Annotations/480p/<sequence>/00000.png
          train.txt
          val.txt

The root can be any directory above that - the locator searches downwards for
``JPEGImages/<resolution>`` and ``Annotations/<resolution>``, so pointing at
``.../DAVIS/2017/trainval``, ``.../DAVIS`` or ``.../DAVIS/2017`` all work.

Annotations are palette PNGs where the colour identifies the object. Rather than
hard-coding the DAVIS palette (which has changed between releases and breaks on
re-encoded copies), the colour -> object mapping is derived from the sequence's
own annotations: black is background, and the remaining colours are sorted and
numbered from 1. Object *ids* therefore may not match the dataset's, which does
not matter here because the harness tracks and scores one object at a time.

The dataset is ~800MB, so nothing is downloaded implicitly. ``download()`` is
there when you want it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .sequences import VideoSequence

__all__ = ["Davis2017", "OFFICIAL_URL", "download"]

OFFICIAL_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
BACKGROUND = (0, 0, 0)


def _find_directory(root: Path, name: str, resolution: str, max_depth: int = 4) -> Path | None:
    """Breadth-first search for ``<name>/<resolution>`` under ``root``."""
    root = root.resolve()
    queue: list[Path] = [root]
    while queue:
        current = queue.pop(0)
        candidate = current / name / resolution
        if candidate.is_dir():
            return candidate
        if len(current.parts) - len(root.parts) < max_depth:
            try:
                queue.extend(child for child in sorted(current.iterdir()) if child.is_dir())
            except (PermissionError, OSError):
                continue
    return None


def _palette_for(annotation_dir: Path) -> dict[tuple[int, int, int], int]:
    """Colour -> object id (1-based) for one sequence, derived from its own PNGs."""
    colours: set[tuple[int, int, int]] = set()
    for path in sorted(annotation_dir.glob("*.png")):
        with Image.open(path) as image:
            if image.mode == "P":
                labels = np.unique(np.asarray(image))
                table = image.getpalette()
                colours.update(tuple(table[int(label) * 3:int(label) * 3 + 3]) for label in labels)
            else:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
                colours.update(tuple(int(c) for c in colour) for colour in np.unique(frame.reshape(-1, 3), axis=0))

    colours.discard(BACKGROUND)
    if not colours:
        raise ValueError(f"{annotation_dir}: every annotation is empty (all background)")
    return {colour: index + 1 for index, colour in enumerate(sorted(colours))}


def _masks_for(annotation_dir: Path, palette: dict[tuple[int, int, int], int]) -> np.ndarray:
    """(T, N, H, W) bool ground truth, one channel per palette entry."""
    paths = sorted(annotation_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no annotations in {annotation_dir}")

    n_objects = max(palette.values())
    with Image.open(paths[0]) as first:
        width, height = first.size

    masks = np.zeros((len(paths), n_objects, height, width), dtype=bool)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            if image.mode == "P":
                if image.size != (width, height):
                    raise ValueError(f"{path}: inconsistent annotation size")
                labels = np.asarray(image)
                table = image.getpalette()
                for label in np.unique(labels):
                    colour = tuple(table[int(label) * 3:int(label) * 3 + 3])
                    object_id = palette.get(colour)
                    if object_id is not None:
                        masks[index, object_id - 1] = labels == label
                continue
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if (frame.shape[1], frame.shape[0]) != (width, height):
            raise ValueError(
                f"{path}: size {(frame.shape[1], frame.shape[0])} != {(width, height)}"
            )
        # One np.unique per frame instead of one comparison per colour: this is the
        # difference between a few seconds and a minute per sequence at 480p.
        unique_colours, inverse = np.unique(
            frame.reshape(-1, 3), axis=0, return_inverse=True
        )
        inverse = inverse.reshape(height, width)
        for label, colour in enumerate(unique_colours):
            object_id = palette.get(tuple(int(c) for c in colour))
            if object_id is not None:
                masks[index, object_id - 1] = inverse == label
    return masks


@dataclass
class Davis2017:
    """A DAVIS 2017 split, loaded one sequence at a time.

    Args:
        root: any directory above ``JPEGImages``/``Annotations``.
        split: ``"train"``, ``"val"`` or ``"all"``. ``test-dev`` has no public
            annotations, so it is not supported for scoring.
        resolution: usually ``"480p"``; ``"Full-Resolution"`` also works.
    """

    root: Path
    split: str = "val"
    resolution: str = "480p"
    strict: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(f"DAVIS root does not exist: {self.root}")

        images = _find_directory(self.root, "JPEGImages", self.resolution)
        annotations = _find_directory(self.root, "Annotations", self.resolution)
        if images is None or annotations is None:
            raise FileNotFoundError(
                f"could not find JPEGImages/{self.resolution} and "
                f"Annotations/{self.resolution} under {self.root}. Pass the directory that "
                "contains them, e.g. .../DAVIS/2017/trainval"
            )
        self._images = images
        self._annotations = annotations
        self._names = self._resolve_names()

    # ------------------------------------------------------------------ splits
    def _resolve_names(self) -> list[str]:
        available = {path.name for path in self._images.iterdir() if path.is_dir()}
        if self.split == "all":
            return sorted(available)

        candidates = list(self.root.rglob(f"{self.split}.txt"))
        # The release contains BOTH 2016 (20 val clips) and 2017 (30 val clips).
        # An arbitrary rglob order can silently run the wrong benchmark.
        candidates.sort(key=lambda path: ("2017" not in path.parts, str(path)))
        for candidate in candidates:
            listed = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            selected = [name for name in listed if name in available]
            if self.strict and len(selected) != len(listed):
                raise ValueError(f"{candidate}: split contains unavailable sequences")
            if selected:
                # Some releases put train and val in the same folder; trust the file,
                # but only for names that are actually present.
                return sorted(selected)

        # No split file found. The standard release keeps train and val sequences
        # in ONE folder and separates them by these files, so without them there is
        # no way to recover the real split - this is a rough 30/rest guess, and the
        # caller is told so rather than being handed a silently wrong benchmark.
        ordered = sorted(available)
        if self.strict:
            raise FileNotFoundError(f"no official {self.split}.txt under {self.root}; refusing to guess a benchmark split")
        if self.split in {"val", "train"}:
            import warnings

            warnings.warn(
                f"no {self.split}.txt found under {self.root}, so the split is a "
                "guess. Download the official split files before reporting numbers.",
                stacklevel=2,
            )
            if len(ordered) > 30:
                return ordered[30:] if self.split == "val" else ordered[:30]
        return ordered

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self):
        for name in self._names:
            yield self.load(name)

    def __repr__(self) -> str:
        preview = ", ".join(self._names[:4])
        if len(self._names) > 4:
            preview += f", +{len(self._names) - 4} more"
        return f"Davis2017({self.root}, split={self.split!r}, {len(self)} sequences: {preview})"

    # ---------------------------------------------------------------- loading
    def load(self, name: str) -> VideoSequence:
        """Load one sequence: annotations in memory, frames left on disk."""
        image_dir = self._images / name
        annotation_dir = self._annotations / name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"no images for sequence {name!r} in {self._images}")
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"no annotations for sequence {name!r} in {self._annotations}")

        palette = _palette_for(annotation_dir)
        masks = _masks_for(annotation_dir, palette)

        n_images = len(sorted(image_dir.glob("*.jpg"))) + len(sorted(image_dir.glob("*.png")))
        if n_images != masks.shape[0]:
            # Silently training on misaligned GT is far worse than refusing to load.
            raise ValueError(
                f"{name}: {n_images} images but {masks.shape[0]} annotations; "
                "the frame and annotation lists disagree"
            )
        image_names = sorted(path.stem for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".png"})
        annotation_names = sorted(path.stem for path in annotation_dir.glob("*.png"))
        if image_names != annotation_names:
            raise ValueError(f"{name}: frame and annotation names disagree")

        return VideoSequence(
            name=name,
            masks=masks,
            frame_dir=image_dir,
            fps=24.0,
            meta={
                "split": self.split,
                "resolution": self.resolution,
                "palette": {f"{c[0]},{c[1]},{c[2]}": i for c, i in palette.items()},
            },
        )


def download(root: str | Path, url: str = OFFICIAL_URL) -> Path:
    """Fetch and unzip the DAVIS 2017 trainval release (~800MB).

    Slow from some networks. If this crawls, grab the same file from a mirror and
    point ``Davis2017(root=...)`` at the unzipped directory instead.
    """
    import urllib.request
    import zipfile

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / Path(url).name

    if not archive.exists():
        print(f"downloading {url}\n  -> {archive} (~800MB)")
        with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
            total = int(response.headers.get("content-length", 0))
            seen = 0
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"\r  {seen / total:6.1%}", end="", flush=True)
        print()

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)
    print(f"extracted into {root}")
    return root
