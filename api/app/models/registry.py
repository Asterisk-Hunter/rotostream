"""Tracker plugin discovery.

Three ways to register a tracker, in increasing order of precedence:

1. **Builtin** — the ``BUILTIN`` mapping below.
2. **Entry point** — a package advertising ``[project.entry-points."rotostream.trackers"]``.
   Lets you ship your model as a separate installed package with no edits here.
3. **Environment** — ``ROTOSTREAM_MODELS="my_key=my_pkg.tracker:MyTracker"``.

Precedence means the env var can shadow a builtin, so you can point the existing
``sam2_memory`` key at your own implementation while iterating.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from .base import TrackerInfo, VideoObjectTracker

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "rotostream.trackers"
ENV_VAR = "ROTOSTREAM_MODELS"

#: ``key -> "module.path:ClassName"``
BUILTIN: dict[str, str] = {
    "naive": "app.models.naive:NaiveColorTracker",
    "sam2_memory": "app.models.sam2_memory:MemoryAttentionTracker",
}

#: Tests and notebooks can register ephemeral trackers here.
_OVERRIDES: dict[str, str] = {}


def register(key: str, spec: str) -> None:
    """Register ``key`` for this process only (mainly useful in tests)."""
    _OVERRIDES[key] = spec


def _env_specs() -> dict[str, str]:
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return {}
    specs: dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk or ":" not in chunk:
            logger.warning("ignoring malformed %s entry: %r", ENV_VAR, chunk)
            continue
        key, spec = chunk.split("=", 1)
        specs[key.strip()] = spec.strip()
    return specs


def _entry_point_specs() -> dict[str, str]:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return {}
    try:
        discovered = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10
        discovered = entry_points().get(ENTRY_POINT_GROUP, [])
    return {ep.name: ep.value for ep in discovered}


def specs() -> dict[str, str]:
    """All known ``key -> spec`` pairs, lowest precedence first."""
    merged = dict(BUILTIN)
    merged.update(_entry_point_specs())
    merged.update(_env_specs())
    merged.update(_OVERRIDES)
    return merged


def load_class(key: str) -> type[VideoObjectTracker]:
    """Import and return the class registered under ``key``."""
    table = specs()
    if key not in table:
        raise KeyError(f"unknown tracker {key!r}; known keys: {', '.join(sorted(table))}")
    spec = table[key]
    if ":" not in spec:
        raise ValueError(f"malformed spec for {key!r}: {spec!r} (expected 'module:Class')")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_name} has no attribute {class_name!r}") from exc
    if not (isinstance(cls, type) and issubclass(cls, VideoObjectTracker)):
        raise TypeError(f"{spec} is not a VideoObjectTracker subclass")
    return cls


def create(key: str, **kwargs: Any) -> VideoObjectTracker:
    """Instantiate the tracker registered under ``key``."""
    return load_class(key)(**kwargs)


def available() -> list[TrackerInfo]:
    """Describe every registered tracker.

    A plugin that fails to import (missing dependency, syntax error) is reported
    with ``implemented=False`` and the error text instead of taking down the API.
    """
    infos: list[TrackerInfo] = []
    for key in sorted(specs()):
        try:
            cls = load_class(key)
            info = cls.info()
            if info.name != key:
                info = TrackerInfo(**{**info.__dict__, "name": key})
            infos.append(info)
        except Exception as exc:  # noqa: BLE001 - surface any import failure to the UI
            logger.warning("tracker %r unavailable: %s", key, exc)
            infos.append(
                TrackerInfo(
                    name=key,
                    description=f"unavailable: {type(exc).__name__}",
                    implemented=False,
                    error=str(exc),
                )
            )
    return infos


def available_keys() -> list[str]:
    return sorted(specs())
