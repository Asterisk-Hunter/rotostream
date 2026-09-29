"""The training objective for the memory stack.

Weights follow the paper (§D.2.2): focal 20, dice 1, IoU regression 1, object
presence 1. Three things about this note are easy to get wrong and are load-bearing:

* The mask term is **focal**, not binary cross-entropy. Focal stops the loss being
  dominated by the (vastly more numerous) background pixels of a small object.
* Only the mask with the **lowest segmentation loss** is supervised. The decoder
  emits several masks for an ambiguous prompt; the correct one is whichever the
  loss itself identifies as the best match, and supervising the rest would punish
  the model for offering a plausible alternative.
* There is **no temporal-consistency term**. Temporality is architectural (memory
  attention), not a penalty on frame-to-frame change. A mask frame's supervision
  must never depend on a frame it has not been shown yet.

When a frame has no ground-truth mask (the object is genuinely absent), the mask
outputs are not supervised at all, but the **presence head always is** -- that is
the only signal that teaches the model to report "the object is gone".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["LOSS_WEIGHTS", "Sam2LossBreakdown", "dice_loss", "sigmoid_focal_loss", "sam2_losses"]

#: ``focal, dice, iou, presence`` -- the paper's §D.2.2 weights.
LOSS_WEIGHTS = {"focal": 20.0, "dice": 1.0, "iou": 1.0, "presence": 1.0}


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss reduced over space only.

    ``logits``/``targets`` are ``(..., h, w)`` and the result keeps every leading
    dimension, so a ``(frames, candidates, h, w)`` input yields one value per frame
    *per candidate mask*. That is the whole point: the caller has to compare the
    candidates against each other, which a scalar-per-mask loss cannot do.
    """
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).flatten(-2).mean(dim=-1)


def dice_loss(probabilities: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice over the last two dims, one value per leading index."""
    probs = probabilities.flatten(-2)
    truth = targets.flatten(-2)
    numerator = 2 * (probs * truth).sum(dim=-1)
    denominator = probs.sum(dim=-1) + truth.sum(dim=-1)
    return 1 - (numerator + eps) / (denominator + eps)


@dataclass
class Sam2LossBreakdown:
    """Total loss plus each supervised term, for logging."""

    total: torch.Tensor
    focal: torch.Tensor
    dice: torch.Tensor
    iou: torch.Tensor
    presence: torch.Tensor
    #: Which candidate mask was supervised on each frame. Kept out of ``to_dict``:
    #: the training harness logs every value it finds as a scalar metric, and an
    #: index is not a metric (averaging it would be meaningless).
    supervised_mask: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Every term, ready for the harness's scalar logging."""
        return {
            "loss": self.total,
            "focal": self.focal,
            "dice": self.dice,
            "iou": self.iou,
            "presence": self.presence,
        }


def sam2_losses(mask_logits: torch.Tensor, iou_predictions: torch.Tensor,
                presence_logits: torch.Tensor, gt_masks: torch.Tensor,
                presence_targets: torch.Tensor) -> Sam2LossBreakdown:
    """Compute the combined objective for one clip.

    Shapes:
        ``mask_logits``      (T, K, h, w)  low-resolution mask logits
        ``iou_predictions``  (T, K)        predicted IoU in [0, 1] (post-sigmoid)
        ``presence_logits``  (T,)          object-presence logit
        ``gt_masks``         (T, h, w)     ground truth **at the same resolution**, bool
        ``presence_targets`` (T,)          whether the object is on screen

    ``gt_masks`` must already be resampled to the logits' resolution: the model
    predicts at stride 4 and comparing against a full-resolution target would
    quietly change the loss scale with the input size.
    """
    if mask_logits.ndim != 4:
        raise ValueError(f"mask_logits must be (T, K, h, w), got {tuple(mask_logits.shape)}")
    if len(gt_masks) != len(mask_logits) or len(presence_targets) != len(mask_logits):
        raise ValueError("mask_logits, gt_masks and presence_targets must agree on the frame count")

    num_frames = mask_logits.shape[0]
    targets = gt_masks.to(mask_logits.dtype).unsqueeze(1)  # (T, 1, h, w)
    has_mask = presence_targets.to(mask_logits.dtype)  # (T,)
    probs = torch.sigmoid(mask_logits)

    # --- mask terms, per candidate mask, per frame ----------------------------
    # The target broadcasts against every candidate, so both terms come out (T, K).
    targets_for_masks = targets.expand_as(mask_logits)
    focal_per_mask = sigmoid_focal_loss(mask_logits, targets_for_masks)
    dice_per_mask = dice_loss(probs, targets_for_masks)

    # Which candidate mask was right? The loss itself decides: supervise the mask
    # with the lowest segmentation loss on that frame. Detached, so the choice is a
    # constant for this step and does not flicker while the weights move.
    segmentation = focal_per_mask + dice_per_mask
    best_index = torch.argmin(segmentation.detach(), dim=-1)  # (T,)
    frame_arange = torch.arange(num_frames, device=mask_logits.device)
    focal = focal_per_mask[frame_arange, best_index]
    dice = dice_per_mask[frame_arange, best_index]

    # Frames with no ground-truth mask carry no mask supervision at all -- only
    # presence. Averaging over the supervised frames keeps the scale independent of
    # how many empty frames a clip happens to contain.
    mask_weight = has_mask
    mask_normaliser = mask_weight.sum().clamp(min=1.0)
    focal = (focal * mask_weight).sum() / mask_normaliser
    dice = (dice * mask_weight).sum() / mask_normaliser

    # --- IoU regression: L1 against the supervised mask's true IoU -------------
    with torch.no_grad():
        chosen = torch.gather(probs, 1, best_index.view(-1, 1, 1, 1).expand(-1, 1, *probs.shape[2:]))
        chosen = (chosen > 0.5).to(mask_logits.dtype)
        union = torch.clamp((chosen + targets).clamp(max=1.0).sum(dim=(1, 2, 3)), min=1e-6)
        intersection = (chosen * targets).sum(dim=(1, 2, 3))
        true_iou = intersection / union
    predicted_iou = iou_predictions[frame_arange, best_index]
    iou = ((predicted_iou - true_iou).abs() * mask_weight).sum() / mask_normaliser

    # --- presence: cross-entropy, supervised on EVERY frame --------------------
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, has_mask.to(presence_logits.dtype), reduction="mean"
    )

    total = (
        LOSS_WEIGHTS["focal"] * focal
        + LOSS_WEIGHTS["dice"] * dice
        + LOSS_WEIGHTS["iou"] * iou
        + LOSS_WEIGHTS["presence"] * presence
    )

    return Sam2LossBreakdown(
        total=total,
        focal=focal.detach(),
        dice=dice.detach(),
        iou=iou.detach(),
        presence=presence.detach(),
        supervised_mask=best_index.detach(),
    )
