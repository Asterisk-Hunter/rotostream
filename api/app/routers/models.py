"""Tracker catalog: what the UI can offer in its model picker."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from ..models import registry
from ..schemas import ModelInfoOut
from ..settings import get_settings

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelInfoOut], summary="List registered trackers")
def list_models() -> list[ModelInfoOut]:
    settings = get_settings()
    out: list[ModelInfoOut] = []
    for info in registry.available():
        out.append(ModelInfoOut(**asdict(info), is_default=info.name == settings.default_model))
    # Default first, then usable plugins, then stubs.
    out.sort(key=lambda m: (not m.is_default, not m.implemented, m.name))
    return out
