"""The from-scratch SAM 2 memory stack.

This package is the model. It is deliberately separate from
``api/app/models/sam2_memory.py`` so that the tracker (contract glue, the part
the application talks to) stays readable and the neural modules stay testable on
their own.

Reused vs built
---------------
**Reused, pretrained, frozen:** the hierarchical Hiera image encoder
(:mod:`transformers` ``Sam2VisionModel``). Training an image backbone is a
different and far more expensive problem, and it is not the paper's
contribution.

**Built here:** the prompt encoder, the memory encoder, the two-queue memory
bank, the memory-attention stack with 2D RoPE, the two-way mask decoder and the
occlusion/presence head.

Module and parameter names deliberately mirror the released SAM 2.1 checkpoint so
the pretrained memory-stack weights can be loaded into :class:`MemoryStack` --
that is how the reimplementation is validated (see
``MemoryStack.load_reference_weights``). See ``docs/MODEL_CONTRACT.md`` for the
full specification and the deliberate directional-memory deviation.
"""

from __future__ import annotations

from .backbone import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_IMAGE_SIZE,
    build_vision_encoder,
    normalize_frames,
    resize_frames,
)
from .losses import Sam2LossBreakdown, sam2_losses
from .mask_decoder import MaskDecoder
from .model import MemoryBank, MemorySlot, MemoryStack, StackCaches
from .modules import MemoryAttention, MemoryEncoder, PromptEncoder

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MODEL_IMAGE_SIZE",
    "MemoryAttention",
    "MemoryBank",
    "MemoryEncoder",
    "MemorySlot",
    "MemoryStack",
    "StackCaches",
    "MaskDecoder",
    "PromptEncoder",
    "Sam2LossBreakdown",
    "build_vision_encoder",
    "normalize_frames",
    "resize_frames",
    "sam2_losses",
]
