"""Shared pytest marks for the API suite.

These live outside ``conftest.py`` on purpose. With the whole repository's tests
run from the root, ``api/tests`` and ``ml/tests`` both contribute a module named
``conftest``, so ``from conftest import requires_ffmpeg`` silently resolves to
whichever one pytest imported first - an import error at best, a test that skips
for the wrong reason at worst. A uniquely named module cannot collide.
"""
from __future__ import annotations

import pytest

from app.video import ffmpeg_available

#: Skip a test when ffmpeg/ffprobe are not on PATH: those tests exercise real
#: encoding and a fake would not tell us anything.
requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH"
)

__all__ = ["requires_ffmpeg"]
