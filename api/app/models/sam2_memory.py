"""YOUR MODEL GOES HERE — the from-scratch memory-attention tracker.

The rest of RotoStream is finished. This file is the seam: implement the five
methods below (plus the two training hooks) and the whole application — upload,
click-to-prompt, propagation, mask overlays, exports, J&F evaluation — starts
using your model with no other edits.

Why the memory system and not the backbone
------------------------------------------
Training a ViT/Hiera image encoder from scratch is a different and far more
expensive problem, and it is not the paper's contribution. So: keep a frozen
pretrained image encoder as the feature extractor (a standard, defensible choice)
and implement the part that actually *is* SAM 2 — the memory mechanism.

What you own (each item is one method you fill in)
--------------------------------------------------
1. ``_encode_image``      — run the frozen backbone over a frame's patches.
2. ``_encode_memory``     — compress a frame + its predicted mask into a memory
                            embedding (masked attention pooling over features).
3. ``_update_memory``     — push into the bank: the prompted frame(s) are always
                            retained, recent frames are kept in a bounded queue,
                            older non-conditional frames are dropped/compressed.
4. ``_memory_attention``  — cross-attention: current-frame tokens attend to the
                            bank. Shape flow:
                                tokens      (B, N, C)
                                memory      (B, M, C)
                                -> (B, N, C)
5. ``_decode_mask``       — mask head: fused features + prompt embeddings
                            -> low-res logits -> (H, W) at source resolution.
6. ``_object_presence``   — occlusion head producing the presence logit that
                            ``FrameResult.object_present`` reports. This is the
                            head that handles the object leaving and re-entering
                            frame; a naive tracker has nothing here.

Rules you must respect (enforced by ``api/tests/test_contract.py``)
-------------------------------------------------------------------
* ``propagate`` must attend only to frames already visited in that direction.
  Forward means indices ``< frame_index``; backward means ``> frame_index``.
  Attending to the future inflates your J&F numbers and is caught by the
  "future leakage" test — run ``python api/scripts/check_model.py sam2_memory``
  before trusting any benchmark result.
* Masks are ``(H, W) bool`` at the **source** frame resolution, not at the
  encoder's internal resolution.
* When ``object_present`` is False the mask must be all-False.
* ``add_prompt`` must both produce the mask for the prompted frame *and* store
  that frame into the memory bank.

Development order that saves time
---------------------------------
1. Implement it, set ``implemented=True`` in ``info()``.
2. ``python api/scripts/check_model.py sam2_memory`` — contract checks.
3. ``python -m ml.evaluate --tracker sam2_memory --dataset synthetic`` — synthetic
   sequences with ground truth, including occlusion and re-entry. Fix
   directionality and off-by-one bugs here, not on real footage.
4. ``python -m ml.train --tracker sam2_memory --dataset davis`` — train the stack.
5. ``python -m ml.evaluate --tracker sam2_memory --dataset davis --split val`` —
   report J & F for the README.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base import (
    Direction,
    FrameResult,
    FrameSource,
    PromptSet,
    TrackerInfo,
    TrainableTracker,
    TrainingSequence,
    resolve_device,
)

#: Where the frozen backbone weights usually live. Shown as a UI hint.
DEFAULT_BACKBONE = "facebook/sam2.1-hiera-tiny"


class MemoryAttentionTracker(TrainableTracker):
    """Memory-attention video object tracker (from-scratch memory stack)."""

    key = "sam2_memory"
    checkpoint_hint = (
        f"Memory stack checkpoint (yours). Frozen backbone: {DEFAULT_BACKBONE}"
    )

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        *,
        memory_bank_size: int = 6,
        num_memory_layers: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        self.backbone_name = backbone
        self.memory_bank_size = int(memory_bank_size)
        self.num_memory_layers = int(num_memory_layers)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)

        self.device = "cpu"
        self.backbone: Any | None = None          # frozen image encoder
        self.memory_encoder: Any | None = None
        self.memory_attention: Any | None = None
        self.mask_decoder: Any | None = None
        self.object_presence_head: Any | None = None

        self._frames: FrameSource | None = None
        self._memory_bank: list[dict[str, Any]] = []
        self._prompt_frame: int | None = None
        self._visited: dict[Direction, list[int]] = {
            Direction.FORWARD: [],
            Direction.BACKWARD: [],
        }

    # ------------------------------------------------------------------ metadata
    @classmethod
    def info(cls) -> TrackerInfo:
        return TrackerInfo(
            name=cls.key,
            description=(
                "From-scratch SAM 2 memory stack: memory encoder, bounded memory "
                "bank, memory attention, mask decoder and occlusion head, over a "
                "frozen pretrained image encoder."
            ),
            uses_memory=True,
            trainable=True,
            # >>> Flip to True once load() works, so the UI starts offering it. <<<
            implemented=False,
            checkpoint_hint=cls.checkpoint_hint,
        )

    # ----------------------------------------------------------------- lifecycle
    def load(self, *, device: str = "auto", checkpoint: str | Path | None = None) -> None:
        raise NotImplementedError(
            "MemoryAttentionTracker.load() is not implemented yet.\n"
            "  1. Build the frozen image encoder + your memory modules and assign them\n"
            "     to self.backbone / self.memory_encoder / self.memory_attention /\n"
            "     self.mask_decoder / self.object_presence_head.\n"
            "  2. Set self.device = resolve_device(device) and .to(self.device).eval().\n"
            "  3. Load your trained memory-stack weights from `checkpoint` if given\n"
            "     (the API passes ROTOSTREAM_* config and the per-request checkpoint).\n"
            "  4. Flip implemented=True in info() so the UI lists this tracker.\n"
            "See api/app/models/sam2_memory.py module docstring for the full checklist."
        )

    def set_video(self, frames: FrameSource) -> None:
        """Attach a video. Precompute per-frame image embeddings here if you can
        afford the memory; ``frames`` is lazy and random-access either way."""
        self._frames = frames
        self.reset()

    def reset(self) -> None:
        self._memory_bank = []
        self._prompt_frame = None
        self._visited = {Direction.FORWARD: [], Direction.BACKWARD: []}

    # ------------------------------------------------------------------ tracking
    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        raise NotImplementedError(
            "add_prompt() unimplemented: encode the prompted frame, fuse the point/box\n"
            "embeddings, decode a mask, then push this frame into the memory bank so\n"
            "later frames can condition on it."
        )

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        raise NotImplementedError(
            "propagate() unimplemented: condition the current frame on the memory bank\n"
            "(visited frames only), decode the mask and presence logit, then store this\n"
            "frame so the next step can attend to it."
        )

    # -------------------------------------------------------------- training hooks
    def trainable_parameters(self) -> Iterable[Any]:
        """Return only YOUR modules — the backbone stays frozen."""
        raise NotImplementedError(
            "Return the parameters of memory_encoder, memory_attention, mask_decoder\n"
            "and object_presence_head. Do not include the frozen backbone."
        )

    def training_step(self, sequence: TrainingSequence) -> dict[str, Any]:
        """Differentiable forward pass over one clip; return at least 'loss'.

        ``sequence.frames`` is (T, H, W, 3) uint8, ``sequence.gt_masks`` is
        (T, H, W) bool and ``sequence.prompts`` seeds frame 0. Extra tensors you
        return alongside 'loss' are logged as metrics.
        """
        raise NotImplementedError(
            "Run the memory stack over the clip and return {'loss': <scalar>, ...}.\n"
            "A useful decomposition: mask BCE/Dice on the tracked frames, plus BCE on\n"
            "the occlusion head against presence labels, plus a temporal-consistency\n"
            "term. Only frames from t-1 and earlier may inform frame t."
        )

    # ------------------------------------------------------------------- internals
    def _encode_image(self, frame: np.ndarray) -> Any:
        """TODO: frozen backbone forward -> multi-scale feature maps / tokens."""
        raise NotImplementedError

    def _encode_memory(self, frame: np.ndarray, mask: np.ndarray) -> Any:
        """TODO: compress (frame features + predicted mask) into one memory embedding."""
        raise NotImplementedError

    def _update_memory(self, frame_index: int, memory: Any, *, prompted: bool = False) -> None:
        """TODO: append to the bank, keeping prompted frames and a bounded recent queue."""
        raise NotImplementedError

    def _memory_attention(self, tokens: Any, memory: Any) -> Any:
        """TODO: cross-attention of current-frame tokens over banked memory."""
        raise NotImplementedError

    def _decode_mask(self, features: Any, prompt_embeddings: Any) -> np.ndarray:
        """TODO: mask head -> (H, W) bool at source resolution."""
        raise NotImplementedError

    def _object_presence(self, features: Any) -> tuple[bool, float]:
        """TODO: occlusion/presence head -> (present, confidence in [0, 1])."""
        raise NotImplementedError
