"""Request/response models for the HTTP API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------- enums
class VideoStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    READY = "ready"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


ExportKind = Literal["alpha_webm", "overlay_mp4", "cutout_zip", "mask_zip", "mask_rle_json", "replace_bg"]


# ------------------------------------------------------------------- prompts
class PointIn(BaseModel):
    """A click in pixel coordinates of the frame it belongs to."""

    model_config = ConfigDict(allow_inf_nan=False)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    positive: bool = True


class BoxIn(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x1")
    @classmethod
    def _non_degenerate_x(cls, v: float, info) -> float:
        if abs(v - info.data.get("x0", 0.0)) < 1.0:
            raise ValueError("box has zero width")
        return v

    @field_validator("y1")
    @classmethod
    def _non_degenerate_y(cls, v: float, info) -> float:
        if abs(v - info.data.get("y0", 0.0)) < 1.0:
            raise ValueError("box has zero height")
        return v


class PromptIn(BaseModel):
    frame_index: int = Field(ge=0)
    points: list[PointIn] = Field(default_factory=list, max_length=128)
    box: BoxIn | None = None

    @field_validator("points")
    @classmethod
    def _need_something(cls, points: list[PointIn], info) -> list[PointIn]:
        if not points and info.data.get("box") is None:
            # box may not be set yet during validation order; checked again below
            return points
        return points


# ------------------------------------------------------------------- requests
class TrackRequest(BaseModel):
    prompts: list[PromptIn] = Field(min_length=1, max_length=900)
    model: str | None = None
    checkpoint: str | None = None
    #: Track forwards and backwards from the prompted frames.
    bidirectional: bool = True
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class PreviewRequest(BaseModel):
    """Mask for a single prompted frame - powers the interactive click feedback."""

    prompt: PromptIn
    model: str | None = None
    checkpoint: str | None = None


class ExportRequest(BaseModel):
    kind: ExportKind
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    #: For kind="replace_bg".
    background: Literal["blur", "black", "white", "green"] = "blur"
    blur_radius: int = Field(default=24, ge=1, le=128)


# ------------------------------------------------------------------ responses
class ModelInfoOut(BaseModel):
    name: str
    description: str = ""
    uses_memory: bool = False
    trainable: bool = False
    implemented: bool = True
    checkpoint_hint: str = ""
    error: str = ""
    is_default: bool = False


class VideoOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    filename: str
    status: VideoStatus
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    n_frames: int = 0
    frame_width: int = 0
    frame_height: int = 0
    size_bytes: int = 0
    created_at: str = ""
    error: str | None = None
    has_audio: bool = False


class UploadOut(BaseModel):
    video: VideoOut
    job_id: str


class FrameScoreOut(BaseModel):
    frame_index: int
    score: float
    object_present: bool
    #: True when this frame was prompted by the user rather than propagated to.
    prompted: bool = False


class SessionOut(BaseModel):
    id: str
    video_id: str
    model: str
    created_at: str
    n_tracked: int = 0
    n_frames: int = 0
    prompt_frames: list[int] = Field(default_factory=list)
    mean_score: float = 0.0
    absent_frames: list[int] = Field(default_factory=list)
    elapsed_s: float = 0.0
    cancelled: bool = False
    status: JobStatus = JobStatus.SUCCEEDED
    error: str | None = None
    scores: list[FrameScoreOut] = Field(default_factory=list)
    memory: dict[str, Any] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: str
    kind: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""
    video_id: str | None = None
    result: dict[str, Any] | None = None


class ExportOut(BaseModel):
    id: str
    video_id: str
    session_id: str | None
    kind: str
    status: JobStatus
    filename: str = ""
    size_bytes: int = 0
    download_url: str = ""
    error: str | None = None


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    device: str
    workspace: str
    ffmpeg: bool
    torch: bool
    default_model: str
