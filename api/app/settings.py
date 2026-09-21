"""Env-driven configuration. Every field is overridable with ``ROTOSTREAM_<NAME>``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    long_side: int = 960
    max_frames: int = 900
    jpeg_quality: int = 92

    # --- uploads ---
    max_upload_mb: int = 512
    allowed_extensions: str = ".mp4,.mov,.m4v,.webm,.mkv,.avi"

    # --- http ---
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
