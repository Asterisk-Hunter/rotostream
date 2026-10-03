"""Health check: surfaces runtime capability so the UI can warn early."""
from __future__ import annotations

import tempfile
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .. import __version__
from ..models.base import resolve_device
from ..models import registry
from ..schemas import HealthOut
from ..settings import get_settings
from ..storage import get_workspace
from ..video import ffmpeg_available

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthOut, summary="Liveness and capabilities")
def health() -> HealthOut:
    settings = get_settings()
    try:
        import torch  # noqa: F401

        has_torch = True
    except Exception:  # noqa: BLE001 - torch is optional
        has_torch = False

    return HealthOut(
        status="ok",
        version=__version__,
        device=resolve_device(settings.device),
        workspace=str(get_workspace().root),
        ffmpeg=ffmpeg_available(),
        torch=has_torch,
        default_model=settings.default_model,
    )


@router.get("/ready", summary="Readiness without loading model weights")
def ready() -> JSONResponse:
    checks = {"ffmpeg": ffmpeg_available(), "storage": False, "tracker": False}
    try:
        with tempfile.TemporaryFile(dir=get_workspace().root) as handle:
            handle.write(b"readiness")
            handle.flush()
        checks["storage"] = True
    except OSError:
        logging.getLogger(__name__).exception("workspace readiness check failed")
    try:
        info = registry.create(get_settings().default_model).info()
        checks["tracker"] = info.implemented and not bool(info.error)
    except Exception:
        logging.getLogger(__name__).exception("tracker readiness check failed")
    healthy = all(checks.values())
    return JSONResponse(status_code=200 if healthy else 503,
                        content={"status": "ready" if healthy else "unavailable", "checks": checks})
