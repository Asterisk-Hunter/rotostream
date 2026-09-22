"""RotoStream ML harness.

Everything around the model, so that building the model is the only job left:

* :mod:`rotostream_ml.metrics` - DAVIS J & F, ported from the official evaluation.
* :mod:`rotostream_ml.synthetic` - deterministic toy sequences with exact ground
  truth, for catching indexing/direction bugs before touching real video.
* :mod:`rotostream_ml.davis` - DAVIS 2017 loader.
* :mod:`rotostream_ml.sequences` - the sequence container both of the above yield.
* :mod:`rotostream_ml.evaluate` - run any registered tracker over a dataset and
  report J & F.
* :mod:`rotostream_ml.leakcheck` - prove the memory bank only reads the past.
* :mod:`rotostream_ml.train` - training loop for ``TrainableTracker`` plugins.

Trackers are looked up through the API's registry, so anything implementing the
contract in ``api/app/models/base.py`` is immediately evaluatable and trainable
without touching this package.
"""

def _ensure_api_importable() -> None:
    """Make the sibling ``api/`` package importable.

    The tracker contract (``app.models.base``) and the plugin registry live in
    the API application, and this harness imports them rather than keeping a
    second copy of the interface that could drift. Nothing installs the API, so
    when the harness is run straight from the repo (``python -m
    rotostream_ml.evaluate``) the sibling directory is added once, here. Under
    pytest, ``ml/tests/conftest.py`` has already done it.
    """
    import importlib.util
    import sys
    from pathlib import Path

    try:
        if importlib.util.find_spec("app.models.base") is not None:
            return
    except (ImportError, ValueError):  # pragma: no cover - namespace conflict
        pass

    candidate = Path(__file__).resolve().parents[2] / "api"
    if (candidate / "app" / "models" / "base.py").is_file():
        sys.path.insert(0, str(candidate))


_ensure_api_importable()

from .metrics import (
    DEFAULT_BOUND_TH,
    DatasetScore,
    FrameScore,
    SequenceScore,
    boundary_f,
    boundary_tolerance,
    frame_jf,
    iou,
    j_and_f,
    score_dataset,
    score_sequence,
    seg2bmap,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BOUND_TH",
    "DatasetScore",
    "FrameScore",
    "SequenceScore",
    "boundary_f",
    "boundary_tolerance",
    "frame_jf",
    "iou",
    "j_and_f",
    "score_dataset",
    "score_sequence",
    "seg2bmap",
    "__version__",
]
