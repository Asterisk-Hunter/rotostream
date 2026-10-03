"""Env-driven configuration. Every field is overridable with ``ROTOSTREAM_<NAME>``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROTOSTREAM_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    workspace_dir: Path = Path("workspace")

    # --- models ---
    default_model: str = "naive"
    device: str = "auto"
    #: Comma-separated "key=module:Class" plugin registrations (mirrors the env var
    #: so entries in .env are honoured even though the registry reads os.environ).
    models: str = ""

    # --- frame extraction ---
    #: Working resolution: extracted frames are scaled so the long side is this.
    #: Masks are produced and exported at this resolution.
    long_side: int = Field(default=960, ge=2, le=4096)
    max_frames: int = Field(default=900, ge=1, le=10000)
    jpeg_quality: int = Field(default=92, ge=1, le=100)
    max_pending_jobs: int = Field(default=16, ge=1, le=1000)
    job_history: int = Field(default=200, ge=1, le=10000)
    preview_cache_size: int = Field(default=1, ge=1, le=8)
    ffmpeg_timeout_s: int = Field(default=600, ge=1, le=3600)
    max_source_pixels: int = Field(default=33177600, ge=1, le=132710400)

    # --- uploads ---
    max_upload_mb: int = Field(default=512, ge=1, le=4096)
    allowed_extensions: str = ".mp4,.mov,.m4v,.webm,.mkv,.avi"

    # --- http ---
    #: Bind address. Consumed by ``scripts/dev.mjs``; uvicorn still takes the
    #: flags it is launched with, so these exist so the port lives in one place.
    host: str = "127.0.0.1"
    #: 8010 rather than 8000 - see the note in .env.example.
    port: int = Field(default=8010, ge=1, le=65535)
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_extension_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",") if ext.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_workspace(self) -> Path:
        """Absolute workspace path, resolved against the ``api/`` directory."""
        base = Path(__file__).resolve().parents[1]
        return self.workspace_dir if self.workspace_dir.is_absolute() else (base / self.workspace_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that change the environment."""
    get_settings.cache_clear()
