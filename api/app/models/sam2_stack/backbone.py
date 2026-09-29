"""The frozen image encoder: the one part that is reused rather than built.

The hierarchical Hiera backbone is pretrained (MAE) and stays frozen. It runs
once per interaction and supplies the *unconditioned* tokens that the memory
stack conditions on -- exactly the role it has in the paper. Nothing here is a
contribution of this project.

``transformers`` provides ``Sam2VisionModel`` (Hiera + the feature-pyramid neck),
so this module is a thin wrapper: frame preparation plus a builder that freezes
the encoder and reports how many parameters were frozen.
"""
from __future__ import annotations

from typing import Any, Iterable

#: The resolution the checkpoint's positional encodings and RoPE grid are built
#: for. Frames are resized to it, and masks are resized back afterwards.
MODEL_IMAGE_SIZE = 1024

#: Standard ImageNet normalisation, which is what the SAM 2 image processor uses.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_frames(pixel_values: Any, mean: Iterable[float] = IMAGENET_MEAN,
                     std: Iterable[float] = IMAGENET_STD) -> Any:
    """Normalise a ``(B, 3, H, W)`` float tensor in [0, 1]."""
    import torch

    m = torch.tensor(tuple(mean), dtype=pixel_values.dtype, device=pixel_values.device)
    s = torch.tensor(tuple(std), dtype=pixel_values.dtype, device=pixel_values.device)
    return (pixel_values - m.view(1, 3, 1, 1)) / s.view(1, 3, 1, 1)


def resize_frames(frames: Any, size: int = MODEL_IMAGE_SIZE) -> Any:
    """``(T, H, W, 3)`` uint8 RGB -> ``(T, 3, size, size)`` normalised float32.

    Bilinear with ``align_corners=False`` and antialiasing on the way down, which
    is what the official processor does. Antialiasing matters: it is the
    difference between a clean downsample and aliased edges that the encoder then
    has to segment.
    """
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(frames):
        frames = torch.as_tensor(frames)
    if frames.dtype != torch.float32:
        frames = frames.to(torch.float32)
    if frames.max() > 1.0:
        frames = frames / 255.0

    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    resized = F.interpolate(
        frames, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    return normalize_frames(resized)


def build_vision_encoder(vision_config: Any) -> Any:
    """Build the frozen Hiera + FPN encoder from a ``Sam2VisionConfig``.

    Every parameter has ``requires_grad`` cleared and the module is put in eval
    mode: the encoder is never trained and its dropout must never fire.
    """
    from transformers import Sam2VisionModel

    encoder = Sam2VisionModel(vision_config)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def frozen_parameter_count(module: Any) -> int:
    """Parameters in ``module`` that will not receive gradients (diagnostics)."""
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)
