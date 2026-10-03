"""RotoStream API entrypoint.

    uvicorn app.main:app --reload --port 8010
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .jobs import JobCapacityError, get_job_manager
from .pipeline import clear_preview_cache
from .models import registry
from .routers import exports, health
from .routers import models as models_router
from .routers import tracking, videos
from .settings import get_settings
from .storage import get_workspace
from .video import ffmpeg_available

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("rotostream")


def apply_model_registrations() -> list[str]:
    """Bridge ``ROTOSTREAM_MODELS`` from ``.env`` into the registry.

    The registry reads ``os.environ`` so that plugin registration also works when
    the variable is exported by a shell; pydantic-settings only populates the
    settings object, so entries coming from ``.env`` are registered explicitly.
    """
    raw = get_settings().models
    if not raw:
        return []
    registered: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk or ":" not in chunk:
            logger.warning("ignoring malformed ROTOSTREAM_MODELS entry: %r", chunk)
            continue
        key, spec = chunk.split("=", 1)
        registry.register(key.strip(), spec.strip())
        registered.append(f"{key.strip()} -> {spec.strip()}")
    return registered


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    workspace = get_workspace()
    workspace.root.mkdir(parents=True, exist_ok=True)
    workspace.recover_interrupted()

    for entry in apply_model_registrations():
        logger.info("registered tracker %s", entry)

    if not ffmpeg_available():
        logger.warning(
            "ffmpeg/ffprobe not found on PATH: uploads and exports will fail. "
            "Install ffmpeg and ensure it is on PATH."
        )

    logger.info("workspace : %s", workspace.root)
    logger.info("trackers  : %s", ", ".join(registry.available_keys()))
    logger.info("default   : %s (device=%s)", settings.default_model, settings.device)
    try:
        yield
    finally:
        get_job_manager().shutdown()
        clear_preview_cache()


app = FastAPI(
    title="RotoStream API",
    version=__version__,
    summary="Rotoscoping studio backend around a pluggable video object tracker.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Exposed so the browser can read mask stats off the preview response.
    expose_headers=["X-Mask-Area", "X-Mask-Ratio"],
)

app.include_router(health.router)
app.include_router(models_router.router)
app.include_router(videos.router)
app.include_router(tracking.router)
app.include_router(exports.router)


@app.exception_handler(JobCapacityError)
async def queue_full(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "5"})


@app.exception_handler(FileNotFoundError)
async def resource_missing(request, exc):
    return JSONResponse(status_code=404, content={"detail": "resource not found"})


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": "RotoStream API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }
