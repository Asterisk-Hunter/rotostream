"""Import paths for the ML tests.

Puts both ``ml/`` (so ``rotostream_ml`` imports) and ``api/`` (so the tracker
registry and the contract are importable) on ``sys.path``, the same way
``api/tests/conftest.py`` does. Keeping both makes it possible to test the
harness against the real plugin registry rather than a stand-in.
"""
from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ML_ROOT.parent / "api"

for path in (str(ML_ROOT), str(API_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure() -> None:
    """Fail loudly if the API package is not importable, rather than per-test."""
    import importlib.util

    if importlib.util.find_spec("app.models.registry") is None:
        raise RuntimeError(
            f"could not import app.models.registry; expected the API at {API_ROOT}"
        )
