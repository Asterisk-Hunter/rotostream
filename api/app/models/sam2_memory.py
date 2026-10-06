"""The from-scratch SAM 2 memory-attention tracker.

Implementing this file is the point of the repository. It is the seam: the API,
the UI, exports and the harness all talk to the five contract methods below and
to nothing else.

Reused vs built
---------------
**Reused, pretrained, frozen:** the hierarchical Hiera image encoder
(:mod:`transformers` ``Sam2VisionModel``, loaded from ``facebook/sam2.1-hiera-*``).
Training a backbone from scratch is a different and far more expensive problem.

**Built here** (see :mod:`app.models.sam2_stack`): the prompt encoder, the memory
encoder, the two-queue memory bank, the 4-block memory attention with 2D RoPE,
the two-way mask decoder and the occlusion/presence head.

Read ``docs/MODEL_CONTRACT.md`` before changing anything here; it carries the
shapes, the paper's numbers and the verification order.

Three corrections worth stating up front, because earlier notes got them wrong
-------------------------------------------------------------------------------
1. **The mask loss is focal, not BCE, and there is no temporal-consistency term.**
   Temporality is *architectural* -- it comes from memory attention, not from a
   penalty on frame-to-frame change.
2. **``_object_presence`` consumes mask-decoder output tokens, not image
   features.** The presence token sits alongside the IoU token and the mask tokens;
   projecting raw image features would be a different (and wrong) head.
3. **``embed_dim=256`` is the model width, but the memory bank runs at 64.** The
   memory encoder projects to 64 channels, and the 256-dim object pointer splits
   into 4 tokens of 64 for cross-attention. A 256-wide bank would be a different
   design.

The memory bank is **directional**: ``FORWARD`` may read only frames with a lower
index, ``BACKWARD`` only a higher one. The paper is bidirectional; this repo is
deliberately stricter so a benchmark number reflects click-then-propagate. Do not
"fix" it (see ``docs/MODEL_CONTRACT.md`` section 4).

Development order that saves time
---------------------------------
1. ``python api/scripts/check_model.py sam2_memory`` -- contract, shapes, leakage.
2. ``python -m rotostream_ml.leakcheck --model sam2_memory --all`` (from ``ml/``)
   -- proves the bank never reads unvisited frames. Do this before trusting a
   number.
3. ``python -m rotostream_ml.evaluate --model sam2_memory --dataset synthetic``
   -- toy clips with exact ground truth, including occlusion and a lookalike
   distractor.
4. ``python -m rotostream_ml.train --model sam2_memory --dry-run`` then
   ``--overfit`` -- verify the training path before spending GPU hours.
5. DAVIS: ``python -m rotostream_ml.train --model sam2_memory --dataset davis
   --root ...`` then
   ``python -m rotostream_ml.evaluate --model sam2_memory --dataset davis
   --root ... --split val --json runs/davis_val.json``.
"""
from __future__ import annotations

import functools
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

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

#: The frozen backbone. Its config defines the whole architecture.
DEFAULT_BACKBONE = "facebook/sam2.1-hiera-tiny"

#: How many encoded frames to keep hot. The pipeline visits each frame once, so
#: this only exists to make repeated propagation of the same frame cheap.
ENCODER_CACHE_SIZE = 6


def _inference(method: Any) -> Any:
    """Run one of the contract's inference methods without tracking gradients.

    Only ``training_step`` needs a graph; every other entry point is inference.
    Wrapping here instead of at each call site also stops the memory bank from
    pinning a per-frame autograd graph, which otherwise grows with the length of
    the clip rather than staying flat. ``torch`` is imported inside the wrapper so
    this module still imports in an environment that has no torch installed.
    """

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import torch

        with torch.no_grad():
            return method(*args, **kwargs)

    return wrapper


class MemoryAttentionTracker(TrainableTracker):
    """Memory-attention video object tracker (from-scratch memory stack)."""

    key = "sam2_memory"
    checkpoint_hint = (
        "Optional memory-stack checkpoint from rotostream_ml.train. "
        f"Frozen backbone: {DEFAULT_BACKBONE}"
    )

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        *,
        memory_bank_size: int | None = None,
        reference_weights: bool = True,
        gradient_checkpointing: bool = True,
        presence_head: bool = True,
        object_pointers: bool = True,
        temporal_position_encoding: bool = True,
    ):
        """``memory_bank_size`` caps how many recent frames are attended to.

        It defaults to the checkpoint's own budget (6 recent frames, the paper's
        default) and exists for the ablation table in the README. The architecture
        itself -- widths, layer counts, the RoPE grid -- comes from the backbone
        config, because that is what the pretrained weights were trained with.
        """
        self.backbone_name = str(backbone)
        self.memory_bank_size = memory_bank_size
        self.reference_weights = bool(reference_weights)
        if memory_bank_size is not None and not 1 <= memory_bank_size <= 8:
            raise ValueError("memory_bank_size must be between 1 and 8")
        self.presence_head = bool(presence_head)
        self.object_pointers = bool(object_pointers)
        self.temporal_position_encoding = bool(temporal_position_encoding)
        #: Recompute activations in the backward pass while training (see
        #: ``sam2_stack.model.MemoryStack.set_gradient_checkpointing``).
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.device = "cpu"
        self.config: Any | None = None
        self.stack: Any | None = None

        self._frames: FrameSource | None = None
        self._bank: Any | None = None
        self._image_positions: Any | None = None
        self._cache: "OrderedDict[int, Any]" = OrderedDict()
        self._prompted: list[int] = []
        self._weight_report: dict[str, list[str]] = {"missing": [], "unexpected": []}

    # ------------------------------------------------------------------ metadata
    @classmethod
    def info(cls) -> TrackerInfo:
        import importlib.util

        missing = [name for name in ("torch", "torchvision", "transformers")
                   if importlib.util.find_spec(name) is None]
        return TrackerInfo(
            name=cls.key,
            description=(
                "From-scratch SAM 2 memory stack: memory encoder, bounded memory bank, "
                "memory attention with 2D RoPE, two-way mask decoder and occlusion head, "
                "over a frozen pretrained Hiera image encoder."
            ),
            uses_memory=True,
            trainable=True,
            # A background-only prompt means "no object on this frame" here, which is
            # a legitimate instruction rather than a contract violation.
            accepts_background_only_prompts=True,
            implemented=not missing,
            error=f"Optional model dependencies missing: {', '.join(missing)}. Install api/requirements-model.txt." if missing else "",
            checkpoint_hint=cls.checkpoint_hint,
        )

    # ----------------------------------------------------------------- lifecycle
    def load(self, *, device: str = "auto", checkpoint: str | Path | None = None) -> None:
        """Build the stack, load weights, and put it on ``device``.

        With no ``checkpoint`` the memory stack is warm-started from the released
        SAM 2.1 weights when ``reference_weights`` is set, and randomly initialised
        otherwise. The frozen encoder is loaded either way.
        """
        import torch
        from transformers import Sam2VideoConfig

        from .sam2_stack import MemoryBank, MemoryStack

        self.device = resolve_device(device)
        self.config = Sam2VideoConfig.from_pretrained(self.backbone_name)
        self.stack = MemoryStack(self.config).to(self.device)

        if checkpoint:
            blob = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            if not isinstance(blob, dict):
                raise ValueError("checkpoint must contain a named tensor state dictionary")
            state = blob.get("weights", blob)
            if isinstance(state, dict) and state.get("kind") == "params":
                raise FileNotFoundError(
                    "this checkpoint stores a flat parameter list; re-save it from a "
                    "tracker that provides state_dict() so the layers can be matched"
                )
            if isinstance(state, dict) and state.get("kind") == "state_dict":
                state = state["state"]
            if not isinstance(state, dict):
                raise ValueError("checkpoint weights must be a named tensor state dictionary")
            if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
                raise ValueError("checkpoint state must map parameter names to tensors")
            # Training checkpoints intentionally omit the frozen vision tower.
            # Restore that tower from the released checkpoint before applying the
            # learned stack, otherwise inference would silently use random Hiera.
            has_encoder = any(name.startswith("vision_encoder.") for name in state)
            if not has_encoder:
                self.stack.load_reference_weights(self.backbone_name)
            missing, unexpected = self.stack.load_state_dict(state, strict=False)
            invalid_missing = [name for name in missing if has_encoder or not name.startswith("vision_encoder.")]
            if invalid_missing or unexpected:
                raise ValueError(
                    f"checkpoint does not match {self.backbone_name}: "
                    f"missing stack keys {invalid_missing[:5]}, unexpected keys {list(unexpected)[:5]}"
                )
            self._weight_report = {"missing": list(missing), "unexpected": list(unexpected)}
        elif self.reference_weights:
            missing, unexpected = self.stack.load_reference_weights(self.backbone_name)
            self._weight_report = {"missing": missing, "unexpected": unexpected}
        else:
            # "From scratch" applies to the contributed memory stack. The
            # reused frozen Hiera tower must still start from pretrained weights.
            self.stack.load_reference_weights(self.backbone_name)
            self.stack.init_weights()

        self.stack.eval()
        self.stack.mask_decoder.presence_head_enabled = self.presence_head
        self.stack.object_pointers_enabled = self.object_pointers
        self.stack.temporal_position_encoding_enabled = self.temporal_position_encoding
        for parameter in self.stack.vision_encoder.parameters():
            parameter.requires_grad = False

        self.memory_bank_size = int(
            self.memory_bank_size or max(1, self.stack.num_maskmem - 1)
        )
        # Budget changes select slots, never change the learned temporal table.
        self._bank = MemoryBank(self.memory_bank_size + 1)
        self._image_positions = self.stack.get_image_wide_positional_embeddings().to(self.device)
        self._cache = OrderedDict()

    def set_video(self, frames: FrameSource) -> None:
        """Attach a video. Encoder outputs are computed lazily, per frame."""
        self._frames = frames
        self.reset()

    def reset(self) -> None:
        """Clear the memory bank, the prompt history and the encoder cache."""
        if self._bank is not None:
            self._bank.clear()
        self._prompted = []
        self._cache = OrderedDict()

    def warmup(self) -> None:
        # Deliberately cheap: the first real frame pays the same cost it would
        # here, and a dummy 1024x1024 forward mostly just warms the GPU caches.
        return None

    # ------------------------------------------------------------------ tracking
    @_inference
    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        """Segment the prompted frame and bank it, unconditioned on memory."""
        self._require_ready()
        index = int(prompts.frame_index)
        if prompts.mask is not None:
            return self._add_mask_prompt(prompts)
        points, labels, boxes = self._prompt_tensors(prompts)

        caches = self._encode_image(index)
        # A prompted frame is not conditioned on memory: the learned "no memory"
        # embedding stands in for an empty bank.
        conditioned = self.stack.trunk_features(caches)
        decoded = self.stack.decode(
            caches, conditioned, self._image_positions,
            points, labels, boxes, None, multimask=True,
        )

        best = int(decoded.iou_scores[0, 0].argmax().item())
        high_res = decoded.high_res_masks[0, 0, best]
        present = bool(decoded.object_score_logits[0, 0, 0] > 0)
        score = float(decoded.iou_scores[0, 0, best])

        self._update_memory(
            index,
            caches=caches,
            high_res_masks=high_res,
            object_score_logits=decoded.object_score_logits[0, 0],
            sam_token=decoded.sam_tokens[0, 0],
            present=present,
            featured=True,
            conditioning=True,
        )
        if index not in self._prompted:
            self._prompted.append(index)

        return FrameResult(
            mask=self._mask_from_logits(high_res, present),
            score=score,
            object_present=present,
            extras={"prompted": True, "prompt_index": index},
        )

    def _add_mask_prompt(self, prompts: PromptSet) -> FrameResult:
        """Use the supplied segmentation directly, as SAM 2's mask-input path does."""
        import torch
        import torch.nn.functional as F

        if prompts.mask.shape != self._frames.shape:
            raise ValueError("mask prompt must match the source frame resolution")
        index = prompts.frame_index
        caches = self._encode_image(index)
        mask = torch.tensor(np.array(prompts.mask), dtype=torch.float32, device=self.device)[None, None]
        mask = F.interpolate(mask, size=(self.stack.image_size, self.stack.image_size), mode="nearest")
        logits = mask[0, 0] * 20.0 - 10.0
        present = bool(prompts.mask.any())
        object_logits = torch.tensor([10.0 if present else -10.0], device=self.device)
        decoded = self.stack.decode(
            caches, self.stack.trunk_features(caches), self._image_positions,
            None, None, None, self.stack.mask_downsample(mask), multimask=False,
        )
        pointer = self.stack.object_pointer(decoded.sam_tokens[0, 0], decoded.object_score_logits[0, 0] > 0)
        if not present:
            pointer = self.stack.no_object_pointer.view(-1)
        self._update_memory(
            index, caches=caches, high_res_masks=logits, object_score_logits=object_logits,
            sam_token=decoded.sam_tokens[0, 0], present=present,
            featured=True, conditioning=True, from_points=False, pointer=pointer,
        )
        if index not in self._prompted:
            self._prompted.append(index)
        return FrameResult(mask=np.array(prompts.mask), score=1.0, object_present=present,
                           extras={"prompted": True, "prompt_index": index, "prompt_mode": "mask"})

    @_inference
    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        """Predict ``frame_index`` from memory that is causal in ``direction``."""
        self._require_ready()
        index = int(frame_index)
        forward = direction is Direction.FORWARD

        memory, memory_positions, num_pointer_tokens, visible = self._memory_attention(index, forward)

        caches = self._encode_image(index)
        conditioned = self.stack.condense_memory(
            caches.features[-1].squeeze(1),
            caches.positions[-1].squeeze(1),
            memory,
            memory_positions,
            num_pointer_tokens,
        )
        decoded = self.stack.decode(
            caches, conditioned, self._image_positions,
            None, None, None, None, multimask=True,
        )

        best = int(decoded.iou_scores[0, 0].argmax().item())
        high_res = decoded.high_res_masks[0, 0, best]
        present = bool(decoded.object_score_logits[0, 0, 0] > 0)
        score = float(decoded.iou_scores[0, 0, best])

        self._update_memory(
            index,
            caches=caches,
            high_res_masks=high_res,
            object_score_logits=decoded.object_score_logits[0, 0],
            sam_token=decoded.sam_tokens[0, 0],
            present=present,
            featured=True,
            conditioning=False,
        )

        return FrameResult(
            mask=self._mask_from_logits(high_res, present),
            score=score,
            object_present=present,
            extras={"memory_frames": visible, "direction": direction.value},
        )

    def memory_state(self) -> dict[str, Any]:
        """Diagnostics for the studio's memory inspector."""
        if self._bank is None:
            return {}
        detail = (
            f"directional bank, up to {self.memory_bank_size} recent frames plus "
            f"{len(self._prompted)} prompted; memory width "
            f"{getattr(self.stack, 'mem_dim', 64)}"
        )
        return {
            "bank_size": len(self._bank),
            "frames": self._bank.frames,
            "prompted": sorted(self._prompted),
            "detail": detail,
        }

    # -------------------------------------------------------------- training hooks
    def trainable_parameters(self) -> Iterable[Any]:
        """Everything except the frozen image encoder.

        Only ``load()`` is required: the training harness asks for the parameter
        list as soon as the model exists, before any video is attached.
        """
        self._require_loaded()
        return [p for p in self.stack.parameters() if p.requires_grad]

    def state_dict(self) -> dict[str, Any]:
        """Trainable state only, so checkpoints stay small.

        ``rotostream_ml.train`` prefers this over a flat parameter list: matching by
        name survives a change in module construction order.
        """
        self._require_loaded()
        return {
            key: value.detach().cpu()
            for key, value in self.stack.state_dict().items()
            if not key.startswith("vision_encoder.")
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._require_loaded()
        merged = self.stack.state_dict()
        merged.update(state_dict)
        self.stack.load_state_dict(merged, strict=False)

    def training_step(self, sequence: TrainingSequence) -> dict[str, Any]:
        """Differentiable forward pass over one clip, returning ``{'loss': tensor, ...}``.

        The clip is walked forward in time and only ever conditions on frames
        already visited, so the training objective cannot be satisfied by reading
        the future. Frames before the first prompt have no memory to condition on
        and are skipped rather than supervised on a guess.
        """
        import torch

        from .frames import ArrayFrameSource
        from .sam2_stack import MemoryBank, sam2_losses

        self._require_loaded()
        stack = self.stack
        stack.eval()  # deterministic; the frozen encoder must never run in train mode
        # A clip is one graph: without recomputation the activations of every frame
        # stay alive until the backward reaches them, which is what puts a 16-frame
        # step past a 6 GB consumer GPU.
        stack.set_gradient_checkpointing(self.gradient_checkpointing)

        frames = np.asarray(sequence.frames)
        gt_masks = np.asarray(sequence.gt_masks).astype(bool)
        self._frames = ArrayFrameSource(frames, fps=24.0)
        self._bank = MemoryBank(stack.num_maskmem)
        self._cache = OrderedDict()

        prompts_by_frame = {int(p.frame_index): p for p in sequence.prompts}
        first_prompt = min(prompts_by_frame) if prompts_by_frame else 0

        logits: list[Any] = []
        ious: list[Any] = []
        presence: list[Any] = []
        targets: list[np.ndarray] = []
        present_targets: list[float] = []

        for index in range(first_prompt, len(frames)):
            caches = self._encode_image(index, detach_encoder=True)
            if index in prompts_by_frame:
                points, labels, boxes = self._prompt_tensors(prompts_by_frame[index])
                conditioned = stack.trunk_features(caches)
                conditioning = True
                visible = []
            else:
                memory, memory_positions, num_pointer_tokens, visible = self._memory_attention(
                    index, True
                )
                if not visible:
                    # Nothing to condition on yet; skip rather than guess.
                    continue
                conditioned = stack.condense_memory(
                    caches.features[-1].squeeze(1), caches.positions[-1].squeeze(1),
                    memory, memory_positions, num_pointer_tokens,
                )
                points = labels = boxes = None
                conditioning = False

            decoded = stack.decode(
                caches, conditioned, self._image_positions,
                points, labels, boxes, None, multimask=True,
            )
            best = int(decoded.iou_scores[0, 0].argmax().item())
            high_res = decoded.high_res_masks[0, 0, best]
            present = decoded.object_score_logits[0, 0, 0] > 0

            self._update_memory(
                index,
                caches=caches,
                high_res_masks=high_res,
                object_score_logits=decoded.object_score_logits[0, 0],
                sam_token=decoded.sam_tokens[0, 0],
                present=bool(present),
                featured=True,
                conditioning=conditioning,
            )

            logits.append(decoded.low_res_masks[0, 0])
            ious.append(decoded.iou_scores[0, 0])
            presence.append(decoded.object_score_logits[0, 0, 0])
            targets.append(gt_masks[index])
            present_targets.append(float(gt_masks[index].any()))

        if not logits:
            raise ValueError("training_step got a clip with no usable frames")

        mask_logits = torch.stack(logits)  # (T, K, h, w)
        height, width = mask_logits.shape[-2:]
        # Ground truth is at full resolution; the loss is computed on the decoder's
        # low-res logits, so the target is downsampled to match -- stack first, then
        # add the single channel dim, so interpolate sees a proper (T, 1, h, w).
        gt_full_res = torch.stack([
            torch.as_tensor(target, device=self.device, dtype=torch.bool).float()
            for target in targets
        ]).unsqueeze(1)
        gt_low_res = torch.nn.functional.interpolate(
            gt_full_res, size=(height, width), mode="nearest"
        ).squeeze(1).bool()

        breakdown = sam2_losses(
            mask_logits=mask_logits,
            iou_predictions=torch.stack(ious),
            presence_logits=torch.stack(presence),
            gt_masks=gt_low_res,
            presence_targets=torch.tensor(present_targets, device=self.device),
        )
        return breakdown.to_dict()

    # ------------------------------------------------------------------- internals
    def _encode_image(self, frame_index: int, detach_encoder: bool = True) -> Any:
        """Frozen backbone forward for one frame, cached, with no encoder gradient."""
        import torch

        cached = self._cache.get(frame_index)
        if cached is not None:
            self._cache.move_to_end(frame_index)
            return cached

        from .sam2_stack import resize_frames

        rgb = self._frames[frame_index]  # type: ignore[index]
        pixel_values = resize_frames(torch.tensor(np.array(rgb, copy=True)), self.stack.image_size)
        pixel_values = pixel_values.to(self.device)

        def run() -> Any:
            return self.stack.encode_frame(pixel_values)

        caches = run() if detach_encoder else torch.no_grad()(run)()
        self._cache[frame_index] = caches
        self._cache.move_to_end(frame_index)
        while len(self._cache) > ENCODER_CACHE_SIZE:
            self._cache.popitem(last=False)
        return caches

    def _memory_attention(self, frame_index: int, forward: bool) -> tuple[Any, Any, int, list[int]]:
        """Gather banked memory visible from ``frame_index`` in ``forward`` direction."""
        return self.stack.build_attention_inputs(self._bank, frame_index, forward)

    def _encode_memory(self, caches: Any, high_res_masks: Any, object_score_logits: Any,
                       from_points: bool) -> tuple[Any, Any]:
        """Compress (frame features + predicted mask) into one banked memory map."""
        return self.stack.encode_memory(caches, high_res_masks, object_score_logits, from_points)

    def _update_memory(self, frame_index: int, *, caches: Any, high_res_masks: Any,
                       object_score_logits: Any, sam_token: Any, present: bool,
                       featured: bool, conditioning: bool, from_points: bool | None = None,
                       pointer: Any = None) -> None:
        """Bank this frame so the next step can attend to it."""
        from .sam2_stack import MemorySlot

        if not featured:
            return
        mask = high_res_masks.view(1, 1, *high_res_masks.shape[-2:])
        features, positions = self._encode_memory(caches, mask, object_score_logits,
                                                 from_points=conditioning if from_points is None else from_points)
        if pointer is None:
            pointer = self.stack.object_pointer(sam_token, object_score_logits > 0)
        self._bank.store(
            MemorySlot(
                frame_index=int(frame_index),
                features=features.squeeze(1),
                positions=positions.squeeze(1),
                pointer=pointer.view(-1),
                conditioning=conditioning,
                present=present,
            )
        )

    def _object_presence(self, object_score_logits: Any) -> tuple[bool, float]:
        """Occlusion/presence head result.

        Consumes the **mask-decoder output token**, not image features: the
        presence token is a sibling of the IoU and mask tokens and is produced by
        the decoder's own MLP head. ``confidence`` is the sigmoid of that logit,
        clamped into ``[0, 1]`` to satisfy the contract.
        """
        import torch

        logit = float(object_score_logits.view(-1)[0])
        confidence = float(torch.sigmoid(torch.tensor(logit)))
        return logit > 0, confidence

    def _prompt_tensors(self, prompts: PromptSet) -> tuple[Any, Any, Any]:
        """Contract prompt -> decoder tensors, in the encoder's 1024-pixel frame."""
        import torch

        size = self.stack.image_size
        width = float(self._frames.width)  # type: ignore[union-attr]
        height = float(self._frames.height)  # type: ignore[union-attr]
        scale_x, scale_y = size / width, size / height

        points = labels = boxes = None
        if prompts.points:
            # Shape (batch, point_batch, n_points, 2) -- the extra level is the
            # "how many masks per point" axis the SAM prompt encoder expects.
            coordinates = [[[[p.x * scale_x, p.y * scale_y] for p in prompts.points]]]
            point_labels = [[[1 if p.positive else 0 for p in prompts.points]]]
            points = torch.tensor(coordinates, dtype=torch.float32, device=self.device)
            labels = torch.tensor(point_labels, dtype=torch.int32, device=self.device)
        if prompts.box is not None:
            box = prompts.box
            boxes = torch.tensor(
                [[[box.x0 * scale_x, box.y0 * scale_y, box.x1 * scale_x, box.y1 * scale_y]]],
                dtype=torch.float32, device=self.device,
            )
        return points, labels, boxes

    def _mask_from_logits(self, high_res_masks: Any, present: bool) -> np.ndarray:
        """Resample a 1024-pixel mask logit map back to the source resolution."""
        import torch
        import torch.nn.functional as F

        if not present:
            # object_present=False requires an all-False mask, per the contract.
            return np.zeros((self._frames.height, self._frames.width), dtype=bool)  # type: ignore[union-attr]

        target = (self._frames.height, self._frames.width)  # type: ignore[union-attr]
        logits = high_res_masks.view(1, 1, *high_res_masks.shape[-2:]).to(torch.float32)
        resized = F.interpolate(logits, size=target, mode="bilinear", align_corners=False)
        return (resized[0, 0] > 0).cpu().numpy()

    def _require_loaded(self) -> None:
        """Weights exist. Enough for the training hooks, which bring their own frames."""
        if self.stack is None:
            raise RuntimeError("load() must be called before tracking")

    def _require_ready(self) -> None:
        """Weights exist *and* a video is attached: the inference path only."""
        self._require_loaded()
        if self._frames is None:
            raise RuntimeError("set_video() must be called before tracking")
