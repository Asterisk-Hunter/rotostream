"""Health check: surfaces runtime capability so the UI can warn early."""
from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..models.base import resolve_device
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
