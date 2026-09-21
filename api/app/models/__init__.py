"""Model boundary: the tracker contract, frame containers and plugin registry."""
from . import registry
from .base import (
    BoxPrompt,
    ContractError,
    Direction,
    FrameResult,
    PointPrompt,
    PromptSet,
    TrackerInfo,
    TrainableTracker,
    TrainingSequence,
    VideoObjectTracker,
    check_frame_result,
    resolve_device,
)
from .frames import ArrayFrameSource, DirectoryFrameSource, FrameSource

__all__ = [
    "registry",
    "BoxPrompt",
    "ContractError",
    "Direction",
    "FrameResult",
    "PointPrompt",
    "PromptSet",
    "TrackerInfo",
    "TrainableTracker",
    "TrainingSequence",
    "VideoObjectTracker",
    "check_frame_result",
    "resolve_device",
    "ArrayFrameSource",
    "DirectoryFrameSource",
    "FrameSource",
]
