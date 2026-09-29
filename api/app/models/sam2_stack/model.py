"""The assembled memory stack: frozen encoder + built-from-scratch memory modules.

This is the video tracking core. It owns the memory bank, the object pointers and
the presence/occlusion handling, and it is the only place that knows the ordering
of a tracking step.

Directional by construction
---------------------------
The paper's memory is bidirectional. This implementation is not: every memory
lookup is filtered by direction, so ``forward`` sees only frames at a lower index
and ``backward`` only frames at a higher index. See ``docs/MODEL_CONTRACT.md``
section 4 for why that deviation is deliberate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from .backbone import build_vision_encoder
from .mask_decoder import MaskDecoder, MaskDecoderOutput
from .modules import (
    FeedForward,
    MemoryAttention,
    MemoryEncoder,
    PromptEncoder,
    RandomFourierPositionalEmbedding,
)

__all__ = ["MemoryBank", "MemoryStack", "MemorySlot", "StackCaches"]


# ------------------------------------------------------------------ the bank
@dataclass
class MemorySlot:
    """One banked frame: its memory map, spatial position code and object pointer."""

    frame_index: int
    features: torch.Tensor  # (H*W, mem_dim)
    positions: torch.Tensor  # (H*W, mem_dim)
    pointer: torch.Tensor  # (hidden_dim,)
    conditioning: bool  # True for a user-prompted frame
    present: bool = True


@dataclass
class StackCaches:
    """Per-frame encoder output, reused across repeated propagations of one frame."""

    features: list[torch.Tensor] = field(default_factory=list)  # (H*W, 1, C) per level
    positions: list[torch.Tensor] = field(default_factory=list)
    high_res: list[torch.Tensor] = field(default_factory=list)  # [stride4, stride8] BCHW
    image_embedding: Optional[torch.Tensor] = None  # (1, C, 64, 64)


class MemoryBank:
    """Two FIFO queues over the frame history, queried **directionally**.

    ``recent`` holds the last ``num_maskmem - 1`` tracked frames; ``prompted``
    holds every user-prompted frame. Prompted frames are never evicted by the
    recent-frame budget -- a click is the strongest signal available and dropping
    it would undo the user's correction.

    Queries never cross the current frame in direction terms: ``FORWARD`` only
    ever returns frames with ``frame_index < current``, ``BACKWARD`` only
    ``> current``. That filter is the whole point of this class.
    """

    def __init__(self, num_maskmem: int = 7):
        self.num_maskmem = int(num_maskmem)
        self._slots: dict[int, MemorySlot] = {}

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def frames(self) -> list[int]:
        return sorted(self._slots)

    def clear(self) -> None:
        self._slots.clear()

    def store(self, slot: MemorySlot) -> None:
        self._slots[slot.frame_index] = slot

    def get(self, frame_index: int) -> Optional[MemorySlot]:
        return self._slots.get(frame_index)

    # ----------------------------------------------------------------- gather
    def gather(self, frame_index: int, forward: bool = True) -> tuple[list[MemorySlot], list[int]]:
        """Return banked slots visible from ``frame_index`` plus their temporal offsets.

        Offsets follow the reference convention: a prompted frame has offset ``0``
        and a recent frame ``d`` steps away has offset ``d``. The offsets index the
        temporal positional-encoding table, so they are part of the checkpoint's
        behaviour, not an implementation detail.
        """
        selected: list[tuple[int, MemorySlot]] = []

        on_visited_side = (lambda idx: idx < frame_index) if forward else (lambda idx: idx > frame_index)

        prompted = [
            slot for slot in self._slots.values()
            if slot.conditioning and on_visited_side(slot.frame_index)
        ]
        prompted.sort(key=lambda slot: slot.frame_index, reverse=not forward)
        for slot in prompted:
            selected.append((0, slot))

        for offset in range(1, self.num_maskmem):
            previous = frame_index - offset if forward else frame_index + offset
            if not (0 <= previous):  # never look outside the clip
                continue
            slot = self._slots.get(previous)
            if slot is not None and not slot.conditioning:
                selected.append((offset, slot))

        # Reject duplicates first-wins, preserving the reference's ordering, then
        # split offsets from slots for the caller.
        seen: set[int] = set()
        slots: list[MemorySlot] = []
        offsets: list[int] = []
        for offset, slot in selected:
            if slot.frame_index in seen:
                continue
            seen.add(slot.frame_index)
            slots.append(slot)
            offsets.append(offset)
        return slots, offsets


# ------------------------------------------------------------------ the stack
class MemoryStack(nn.Module):
    """Frozen Hiera encoder plus the memory modules, mirroring the SAM 2.1 layout."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        vision = config.vision_config
        self.hidden_dim = int(vision.fpn_hidden_size)
        self.mem_dim = int(config.memory_encoder_output_channels)
        self.num_maskmem = int(config.num_maskmem)
        self.image_size = int(config.image_size)
        self.backbone_feature_sizes = [tuple(size) for size in vision.backbone_feature_sizes]
        self.max_object_pointers = int(config.max_object_pointers_in_encoder)

        # --- reused, frozen -------------------------------------------------
        self.vision_encoder = build_vision_encoder(vision)

        # --- built here -----------------------------------------------------
        self.shared_image_embedding = RandomFourierPositionalEmbedding(
            config.prompt_encoder_config.hidden_size, config.prompt_encoder_config.scale
        )
        self.prompt_encoder = PromptEncoder(
            hidden_size=config.prompt_encoder_config.hidden_size,
            image_size=config.prompt_encoder_config.image_size,
            patch_size=config.prompt_encoder_config.patch_size,
            mask_input_channels=config.prompt_encoder_config.mask_input_channels,
            num_point_embeddings=config.prompt_encoder_config.num_point_embeddings,
            layer_norm_eps=config.prompt_encoder_config.layer_norm_eps,
            scale=config.prompt_encoder_config.scale,
            hidden_act=config.prompt_encoder_config.hidden_act,
        )
        decoder = config.mask_decoder_config
        self.mask_decoder = MaskDecoder(
            hidden_size=decoder.hidden_size,
            num_multimask_outputs=decoder.num_multimask_outputs,
            num_hidden_layers=decoder.num_hidden_layers,
            num_attention_heads=decoder.num_attention_heads,
            mlp_dim=decoder.mlp_dim,
            attention_downsample_rate=decoder.attention_downsample_rate,
            iou_head_depth=decoder.iou_head_depth,
            iou_head_hidden_dim=decoder.iou_head_hidden_dim,
            dynamic_multimask_via_stability=decoder.dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=decoder.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=decoder.dynamic_multimask_stability_thresh,
        )
        self.memory_attention = MemoryAttention(
            hidden_size=config.memory_attention_hidden_size,
            num_layers=config.memory_attention_num_layers,
            num_attention_heads=config.memory_attention_num_attention_heads,
            downsample_rate=config.memory_attention_downsample_rate,
            feed_forward_hidden_size=config.memory_attention_feed_forward_hidden_size,
            dropout=config.memory_attention_dropout,
            feed_forward_hidden_act=config.memory_attention_feed_forward_hidden_act,
            rope_feat_sizes=tuple(config.memory_attention_rope_feat_sizes),
            rope_theta=config.memory_attention_rope_theta,
        )
        self.memory_encoder = MemoryEncoder(
            hidden_size=config.memory_encoder_hidden_size,
            output_channels=config.memory_encoder_output_channels,
            mask_downsampler_embed_dim=config.mask_downsampler_embed_dim,
            mask_downsampler_kernel_size=config.mask_downsampler_kernel_size,
            mask_downsampler_stride=config.mask_downsampler_stride,
            mask_downsampler_padding=config.mask_downsampler_padding,
            mask_downsampler_total_stride=config.mask_downsampler_total_stride,
            mask_downsampler_hidden_act=config.mask_downsampler_hidden_act,
            fuser_num_layers=config.memory_fuser_num_layers,
            fuser_embed_dim=config.memory_fuser_embed_dim,
            fuser_intermediate_dim=config.memory_fuser_intermediate_dim,
            fuser_kernel_size=config.memory_fuser_kernel_size,
            fuser_padding=config.memory_fuser_padding,
            fuser_layer_scale_init_value=config.memory_fuser_layer_scale_init_value,
            fuser_hidden_act=config.memory_fuser_hidden_act,
        )

        # --- memory bookkeeping parameters ----------------------------------
        self.no_memory_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_memory_positional_encoding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.memory_temporal_positional_encoding = nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )
        self.no_object_pointer = nn.Parameter(torch.zeros(1, self.hidden_dim))

        # Downsamples a mask prompt to stride 4 and rescales it to SAM logit range.
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.object_pointer_proj = FeedForward(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)

        self.enable_temporal_pos_encoding_for_object_pointers = bool(
            config.enable_temporal_pos_encoding_for_object_pointers
        )
        if self.enable_temporal_pos_encoding_for_object_pointers:
            self.temporal_positional_encoding_projection_layer = nn.Linear(self.hidden_dim, self.mem_dim)

        self.enable_occlusion_spatial_embedding = bool(config.enable_occlusion_spatial_embedding)
        if self.enable_occlusion_spatial_embedding:
            self.occlusion_spatial_embedding_parameter = nn.Parameter(torch.zeros(1, self.mem_dim))

        self.sigmoid_scale_for_mem_enc = float(config.sigmoid_scale_for_mem_enc)
        self.sigmoid_bias_for_mem_enc = float(config.sigmoid_bias_for_mem_enc)
        self.multimask_output_for_tracking = bool(config.multimask_output_for_tracking)
        self.multimask_min_pt_num = int(config.multimask_min_pt_num)
        self.multimask_max_pt_num = int(config.multimask_max_pt_num)

        # The prompt encoder's positional buffer is tied to the image-wide one.
        self.prompt_encoder.shared_embedding.positional_embedding = (
            self.shared_image_embedding.positional_embedding
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade compute for activation memory while training.

        Off for inference (there is no graph to save) and switched on by the
        tracker's ``training_step``. See ``modules.run_block`` for why the clip
        length, not the model size, is what makes a training step expensive.
        """
        for module in (self.memory_attention, self.memory_encoder, self.mask_decoder.transformer):
            module.gradient_checkpointing = bool(enabled)

    # ------------------------------------------------------------- weight init
    def init_weights(self) -> None:
        """Sensible init for training from scratch (used when no checkpoint exists).

        Deliberately zeroes the memory bookkeeping parameters: they are additive
        embeddings, so zero is the neutral starting point. The fuser's layer scales
        start at zero so the residual branch is initially an identity, which is the
        ConvNeXt convention and keeps an untrained stack stable.
        """
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        from .modules import MemoryFuserBlock

        for module in self.modules():
            if isinstance(module, MemoryFuserBlock):
                nn.init.zeros_(module.scale)
        with torch.no_grad():
            self.no_memory_embedding.zero_()
            self.no_memory_positional_encoding.zero_()
            self.memory_temporal_positional_encoding.zero_()
            self.no_object_pointer.zero_()
            if self.enable_occlusion_spatial_embedding:
                self.occlusion_spatial_embedding_parameter.zero_()
        # The frozen encoder must survive init untouched.
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False

    def load_reference_weights(self, backbone_id: str) -> tuple[list[str], list[str]]:
        """Warm-start from the released SAM 2.1 checkpoint.

        Returns ``(missing, unexpected)`` key lists so the caller can report exactly
        what did and did not load. The vision encoder is loaded here too; it is the
        same checkpoint, so this is also how the frozen encoder gets its weights.
        """
        from transformers import Sam2VideoModel

        reference = Sam2VideoModel.from_pretrained(backbone_id)
        state = reference.state_dict()
        missing, unexpected = self.load_state_dict(state, strict=False)
        del reference
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        return list(missing), list(unexpected)

    # ------------------------------------------------------------ image encode
    @torch.no_grad()
    def get_image_wide_positional_embeddings(self) -> torch.Tensor:
        """Position codes for the trunk's grid, ``(1, C, 64, 64)``."""
        size = self.prompt_encoder.image_embedding_size
        dtype = self.shared_image_embedding.positional_embedding.dtype
        device = self.shared_image_embedding.positional_embedding.device
        grid = torch.ones(size, device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size[0]
        x_embed = x_embed / size[1]
        embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return embedding.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def encode_frame(self, pixel_values: torch.Tensor) -> StackCaches:
        """Run the frozen encoder once and prepare every view the decoder needs."""
        outputs = self.vision_encoder(pixel_values)
        levels = list(outputs.fpn_hidden_states)
        position = list(outputs.fpn_position_encoding)

        # Levels 0 and 1 are the stride-4 / stride-8 skip connections, projected to
        # the decoder's upsampling widths.
        high_res_s0 = self.mask_decoder.conv_s0(levels[0])
        high_res_s1 = self.mask_decoder.conv_s1(levels[1])

        features = [level.flatten(2).permute(2, 0, 1) for level in levels]
        positions = [pos.flatten(2).permute(2, 0, 1) for pos in position]
        return StackCaches(
            features=features,
            positions=positions,
            high_res=[high_res_s0, high_res_s1],
            image_embedding=levels[-1],
        )

    # ------------------------------------------------------------- one frame
    def trunk_features(self, caches: StackCaches) -> torch.Tensor:
        """The conditioned trunk map for a *prompted* frame, ``(1, C, 64, 64)``.

        A prompted frame is not conditioned on memory; the learned ``no memory``
        embedding stands in for an empty bank.
        """
        # The feature sequence is channel-fastest, so the (C, H, W) map must be
        # built with a permute -- a plain transpose+view would scramble channels
        # into spatial positions.
        height, width = self.backbone_feature_sizes[-1]
        flat = caches.features[-1] + self.no_memory_embedding
        return flat.permute(1, 2, 0).view(1, self.hidden_dim, height, width)

    def condense_memory(self, conditioned_source: torch.Tensor, source_positions: torch.Tensor,
                        memory: torch.Tensor, memory_positions: torch.Tensor,
                        num_pointer_tokens: int) -> torch.Tensor:
        """Condition a frame's trunk features on banked memory, ``(1, C, 64, 64)``."""
        height, width = self.backbone_feature_sizes[-1]
        conditioned = self.memory_attention(
            features=conditioned_source.unsqueeze(0),
            memory=memory.unsqueeze(0),
            features_pos=source_positions.unsqueeze(0),
            memory_pos=memory_positions.unsqueeze(0),
            num_object_pointer_tokens=num_pointer_tokens,
        )
        return conditioned.transpose(1, 2).view(1, self.hidden_dim, height, width)

    # ------------------------------------------------------------ memory build
    def encode_memory(self, caches: StackCaches, high_res_masks: torch.Tensor,
                      object_score_logits: torch.Tensor, from_points: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a predicted mask + frame features into one banked memory map.

        ``from_points`` switches the mask to a hard binary choice, which is what a
        user's click deserves; tracked frames use the soft sigmoid instead.
        """
        pixel_features = caches.image_embedding

        if from_points and not self.training:
            mask = (high_res_masks > 0).to(high_res_masks.dtype)
        else:
            mask = torch.sigmoid(high_res_masks)
        mask = mask * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc

        features, positions = self.memory_encoder(pixel_features, mask)

        # A learned embedding marks "this frame had no object", so the bank records
        # the absence instead of only ever recording a mask.
        if self.enable_occlusion_spatial_embedding:
            appearing = (object_score_logits > 0).float()
            features = features + (1 - appearing[..., None]) * self.occlusion_spatial_embedding_parameter[
                ..., None, None
            ].expand_as(features)

        features = features.flatten(2).permute(2, 0, 1)
        positions = positions.flatten(2).permute(2, 0, 1)
        return features, positions

    def object_pointer(self, sam_token: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Project the chosen mask token into an object pointer, honouring absence."""
        pointer = self.object_pointer_proj(sam_token)
        appearing = present.to(pointer.dtype)
        return appearing * pointer + (1 - appearing) * self.no_object_pointer

    # ------------------------------------------------------------- attention in
    def build_attention_inputs(self, bank: MemoryBank, frame_index: int, forward: bool,
                               ) -> tuple[torch.Tensor, torch.Tensor, int, list[int]]:
        """Gather banked memory, its position codes and the object pointers.

        Returns the concatenated memory and position codes (pointers last, so the
        RoPE exclusion is a simple suffix), the number of pointer tokens, and the
        frame indices that contributed.
        """
        slots, offsets = bank.gather(frame_index, forward=forward)
        if not slots:
            raise ValueError("no memory is visible from this frame; prompt before propagating")

        memory = torch.cat([slot.features for slot in slots], dim=0)
        # The temporal table is stored flat per offset; the spatial code is added to
        # it so the bank's position code carries both where and when.
        memory_positions = torch.cat(
            [
                slot.positions
                + self.memory_temporal_positional_encoding[offset - 1].reshape(-1)
                for slot, offset in zip(slots, offsets)
            ],
            dim=0,
        )

        pointers = self._collect_pointers(slots, offsets)
        pointer_positions = self._pointer_positions(offsets)
        memory = torch.cat([memory, pointers], dim=0)
        memory_positions = torch.cat([memory_positions, pointer_positions], dim=0)
        return memory, memory_positions, pointers.shape[0], [slot.frame_index for slot in slots]

    def _collect_pointers(self, slots: list[MemorySlot], offsets: list[int]) -> torch.Tensor:
        """256-dim pointers, split into 4x64 tokens to match the 64-dim bank width."""
        pointers = torch.stack([slot.pointer for slot in slots], dim=0)  # (N, 256)
        if self.mem_dim < pointers.shape[-1]:
            splits = pointers.shape[-1] // self.mem_dim
            pointers = pointers.view(len(slots), splits, self.mem_dim).permute(1, 0, 2).reshape(-1, self.mem_dim)
        return pointers

    def _pointer_positions(self, offsets: list[int]) -> torch.Tensor:
        """1D sinusoidal code of the temporal offset, projected to the bank width.

        Object pointers carry no spatial position, so their only positional signal
        is *when* the frame was, which is exactly what this encodes.
        """
        count = len(offsets)
        if not self.enable_temporal_pos_encoding_for_object_pointers:
            zeros = torch.zeros(count * 4, self.mem_dim, dtype=self.no_object_pointer.dtype,
                                device=self.no_object_pointer.device)
            return zeros

        max_temporal_diff = float(max(self.max_object_pointers - 1, 1))
        diffs = torch.tensor(offsets, dtype=torch.float32, device=self.no_object_pointer.device)
        normalized = diffs / max_temporal_diff

        half = self.hidden_dim // 2
        dim_t = torch.arange(half, dtype=torch.float32, device=normalized.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half)
        embeddings = normalized.unsqueeze(-1) / dim_t
        sine = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)  # (N, 256)
        projected = self.temporal_positional_encoding_projection_layer(sine)  # (N, 64)
        # Each pointer splits into `splits` tokens, so its position code repeats.
        splits = self.hidden_dim // self.mem_dim
        return projected.repeat_interleave(splits, dim=0)

    # ------------------------------------------------------------------ decode
    def decode(self, caches: StackCaches, conditioned: torch.Tensor, image_positions: torch.Tensor,
               points: Optional[torch.Tensor], labels: Optional[torch.Tensor],
               boxes: Optional[torch.Tensor], masks: Optional[torch.Tensor],
               multimask: bool) -> MaskDecoderOutput:
        sparse, dense = self.prompt_encoder(points=points, labels=labels, boxes=boxes, masks=masks)
        if sparse is None:
            # No prompt at all still needs a token sequence; the padded "not a
            # point" token is what the decoder was trained to expect.
            batch = conditioned.shape[0]
            sparse = self.prompt_encoder(
                points=torch.zeros(batch, 1, 1, 2, dtype=conditioned.dtype, device=conditioned.device),
                labels=-torch.ones(batch, 1, 1, dtype=torch.int32, device=conditioned.device),
                boxes=None,
                masks=None,
            )[0]

        image_size = self.image_size
        return self.mask_decoder(
            image_embeddings=conditioned,
            image_positional_embeddings=image_positions,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            high_resolution_features=caches.high_res,
            image_size=image_size,
        )
