"""Neural modules for the memory stack: layers, prompt encoder, memory encoder
and memory attention.

Parameter and submodule names match the released SAM 2.1 checkpoint, which is
what lets pretrained weights be loaded into this implementation. The maths
follows the paper (see ``docs/MODEL_CONTRACT.md``); the code is written here.

Why the names matter
--------------------
Loading ``facebook/sam2.1-hiera-tiny`` into :class:`~.model.MemoryStack` is this
reimplementation's correctness check. Renaming ``cross_attn_image`` or
``mask_downsampler`` would silently break that check, so the names are part of
the contract with the reference checkpoint, not an accident.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "ACTIVATIONS",
    "Attention",
    "ConvNorm",
    "FeedForward",
    "MaskDownSampler",
    "MaskDownSamplerLayer",
    "MaskEmbedding",
    "MemoryAttention",
    "MemoryAttentionLayer",
    "MemoryEncoder",
    "MemoryFuser",
    "MemoryFuserBlock",
    "PromptEncoder",
    "RandomFourierPositionalEmbedding",
    "RoPEAttention",
    "SinePositionEmbedding",
    "VisionRotaryEmbedding",
]

ACTIVATIONS: dict[str, Any] = {"relu": nn.ReLU, "gelu": nn.GELU}


def run_block(module: nn.Module, *args: Any, recompute: bool = False) -> Any:
    """Call ``module``, optionally re-running it in the backward pass.

    Training a clip is one long graph: every frame's mask decoder and every
    memory-encoder pass stays alive until the backward reaches it. A single
    stride-2 convolution over a 1024x1024 mask is already a quarter of a gigabyte
    of activation, and there is one of those per frame, so the clip length -- not
    the model -- is what decides whether a step fits on a consumer GPU.
    Checkpointing trades compute for activation memory, which is the right trade
    when the alternative is not being able to train at all. It is off by default
    and only ever switched on for training; inference has no graph to save.
    """
    if not recompute:
        return module(*args)
    return checkpoint(module, *args, use_reentrant=False)


# --------------------------------------------------------------------- layers
class ConvNorm(nn.LayerNorm):
    """LayerNorm that also accepts channels-first ``(B, C, H, W)`` tensors.

    The convolution stacks in the memory encoder are channels-first; the fuser's
    pointwise stages are channels-last. This lets one module serve both without a
    permute at every call site.
    """

    def __init__(self, normalized_shape: Any, *, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__(normalized_shape, eps=eps)
        if data_format not in {"channels_last", "channels_first"}:
            raise NotImplementedError(f"unsupported data_format {data_format!r}")
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            return features.permute(0, 3, 1, 2)
        return super().forward(features)


class SinePositionEmbedding(nn.Module):
    """Sinusoidal 2D position embedding, generalised from *Attention Is All You Need*.

    ``normalize=True`` rescales the coordinates to ``2*pi`` over the feature map,
    matching the reference and therefore the trained weights.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, shape: torch.Size, device: Any, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros((shape[0], shape[2], shape[3]), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class FeedForward(nn.Module):
    """``num_layers``-deep MLP: in -> hidden -> ... -> out."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 activation: str = "relu", sigmoid_output: bool = False):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.num_layers = num_layers
        self.activation = ACTIVATIONS[activation]()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.proj_in(hidden_states))
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))
        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = torch.sigmoid(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Multi-head attention with a projection bottleneck (the SAM decoder's block).

    ``downsample_rate`` shrinks the internal width so the two-way transformer can
    run cheaply at 64x64 tokens.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.internal_dim = hidden_size // downsample_rate
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.internal_dim // num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(hidden_size, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_similarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, queries, _ = query.shape
        keys = key.shape[1]

        q = self.q_proj(query).view(batch, queries, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, keys, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, keys, self.num_attention_heads, self.head_dim).transpose(1, 2)

        weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_similarity is not None:
            weights = weights + attention_similarity
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(batch, queries, self.internal_dim)
        return self.o_proj(out)


# ------------------------------------------------------------ prompt encoder
class RandomFourierPositionalEmbedding(nn.Module):
    """Fourier features of a normalised coordinate. The SAM prompt position code."""

    def __init__(self, hidden_size: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, hidden_size // 2)))

    def forward(self, input_coords: torch.Tensor, input_shape: Optional[tuple[int, int]] = None) -> torch.Tensor:
        coordinates = input_coords.clone().to(self.positional_embedding.dtype)
        if input_shape is not None:
            coordinates[..., 0] = coordinates[..., 0] / input_shape[1]
            coordinates[..., 1] = coordinates[..., 1] / input_shape[0]
        coordinates = 2 * coordinates - 1
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * math.pi * coordinates
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class MaskEmbedding(nn.Module):
    """Dense mask prompt: 1 channel at stride 4 -> ``hidden_size`` channels."""

    def __init__(self, hidden_size: int, mask_input_channels: int, layer_norm_eps: float = 1e-6,
                 hidden_act: str = "gelu"):
        super().__init__()
        self.mask_input_channels = mask_input_channels // 4
        self.activation = ACTIVATIONS[hidden_act]()
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(mask_input_channels, hidden_size, kernel_size=1)
        self.layer_norm1 = ConvNorm(self.mask_input_channels, eps=layer_norm_eps, data_format="channels_first")
        self.layer_norm2 = ConvNorm(self.mask_input_channels * 4, eps=layer_norm_eps, data_format="channels_first")

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.layer_norm1(self.conv1(masks)))
        hidden_states = self.activation(self.layer_norm2(self.conv2(hidden_states)))
        return self.conv3(hidden_states)


class PromptEncoder(nn.Module):
    """Points and boxes -> sparse embeddings; an optional mask -> dense embeddings.

    This follows SAM's design (a random Fourier code per point plus a learned
    embedding per prompt type) and is re-implemented here rather than reused; it is
    not one of the paper's contributions.
    """

    def __init__(self, hidden_size: int = 256, image_size: int = 1024, patch_size: int = 16,
                 mask_input_channels: int = 16, num_point_embeddings: int = 4,
                 layer_norm_eps: float = 1e-6, scale: float = 1.0, hidden_act: str = "gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_image_size = image_size
        self.shared_embedding = RandomFourierPositionalEmbedding(hidden_size, scale)
        self.mask_embed = MaskEmbedding(hidden_size, mask_input_channels, layer_norm_eps, hidden_act)
        self.no_mask_embed = nn.Embedding(1, hidden_size)

        self.image_embedding_size = (image_size // patch_size, image_size // patch_size)
        self.mask_input_size = (4 * image_size // patch_size, 4 * image_size // patch_size)

        self.point_embed = nn.Embedding(num_point_embeddings, hidden_size)
        self.not_a_point_embed = nn.Embedding(1, hidden_size)

    # Padding is what keeps a variable number of clicks in one tensor: label -1
    # marks a slot to be ignored.
    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5
        if pad:
            points = F.pad(points, (0, 0, 0, 1), mode="constant", value=0)
            labels = F.pad(labels, (0, 1), mode="constant", value=-1)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        point_embedding = torch.where(
            labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding
        )
        point_embedding = torch.where(
            labels[..., None] != -10, point_embedding, torch.zeros_like(point_embedding)
        )
        return point_embedding + self.point_embed(labels.clamp(min=0)) * (labels >= 0).unsqueeze(-1)

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.view(*boxes.shape[:2], 2, 2)
        coords = F.pad(coords, (0, 0, 0, 1), mode="constant", value=0)
        corner = self.shared_embedding(coords, (self.input_image_size, self.input_image_size))
        corner[:, :, 0, :] = corner[:, :, 0, :] + self.point_embed.weight[2]
        corner[:, :, 1, :] = corner[:, :, 1, :] + self.point_embed.weight[3]
        corner[:, :, 2, :] = self.not_a_point_embed.weight.expand_as(corner[:, :, 2, :])
        return corner

    def forward(self, points: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None, masks: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse = None
        batch_size = 1
        if points is not None:
            if labels is None:
                raise ValueError("point prompts need labels")
            batch_size = points.shape[0]
            sparse = self._embed_points(points, labels, pad=(boxes is None))
        if boxes is not None:
            batch_size = boxes.shape[0]
            box_embeddings = self._embed_boxes(boxes)
            sparse = box_embeddings if sparse is None else torch.cat([sparse, box_embeddings], dim=2)
        if masks is not None:
            dense = self.mask_embed(masks)
        else:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse, dense


# ----------------------------------------------------------- memory attention
def _rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    """Pairwise 2D rotation of the last dimension (not the Llama half-split)."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class VisionRotaryEmbedding(nn.Module):
    """Fixed axial 2D RoPE table for a ``feat_h x feat_w`` feature grid.

    The grid is fixed because the encoder always runs at one resolution, so the
    table is precomputed once instead of per forward pass.
    """

    def __init__(self, hidden_size: int, feat_sizes: tuple[int, int], num_heads: int = 1,
                 downsample_rate: int = 1, theta: float = 10000.0):
        super().__init__()
        dim = hidden_size // (downsample_rate * num_heads)
        if dim % 4 != 0:
            raise ValueError("head dim must be divisible by 4 for axial RoPE")
        end_x, end_y = feat_sizes
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))

        flat = torch.arange(end_x * end_y, dtype=torch.long)
        x_positions = flat % end_x
        y_positions = torch.div(flat, end_x, rounding_mode="floor")
        inv_freq = torch.cat([torch.outer(x_positions, freqs).float(),
                              torch.outer(y_positions, freqs).float()], dim=-1)
        inv_freq = inv_freq.repeat_interleave(2, dim=-1)
        # Buffers, not parameters: they are recomputed from config, never trained.
        self.register_buffer("rope_embeddings_cos", inv_freq.cos(), persistent=False)
        self.register_buffer("rope_embeddings_sin", inv_freq.sin(), persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rope_embeddings_cos, self.rope_embeddings_sin


def apply_rotary_2d(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                    num_k_exclude_rope: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and the spatial keys; leave the trailing pointer keys alone.

    Object-pointer tokens are appended to the end of the memory sequence and have
    no spatial correspondence, so they are excluded from RoPE by construction.
    """
    k_rot = key[..., : key.shape[-2] - num_k_exclude_rope, :]
    k_pass = key[..., key.shape[-2] - num_k_exclude_rope:, :]

    q_embed = query.float()
    q_embed = (q_embed * cos) + (_rotate_pairwise(q_embed) * sin)

    if k_rot.shape[-2] == 0:
        return q_embed.type_as(query), torch.cat([k_rot, k_pass], dim=-2)

    if k_rot.shape[-2] % cos.shape[-2] != 0:
        raise ValueError(
            f"memory key length {k_rot.shape[-2]} is not a whole number of "
            f"{cos.shape[-2]}-token frames; the bank and the RoPE grid disagree"
        )
    repeats = k_rot.shape[-2] // cos.shape[-2]
    cos_k = cos.repeat(1, 1, repeats, 1) if repeats > 1 else cos
    sin_k = sin.repeat(1, 1, repeats, 1) if repeats > 1 else sin

    k_embed = k_rot.float()
    k_embed = (k_embed * cos_k) + (_rotate_pairwise(k_embed) * sin_k)
    k_embed = torch.cat([k_embed.type_as(key), k_pass], dim=-2)
    return q_embed.type_as(query), k_embed


class RoPEAttention(nn.Module):
    """Attention with axial 2D rotary position encoding on queries and keys."""

    def __init__(self, hidden_size: int = 256, num_attention_heads: int = 1,
                 downsample_rate: int = 1, kv_in_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.internal_dim = hidden_size // downsample_rate
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.internal_dim // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.kv_in_dim = hidden_size if kv_in_dim is None else kv_in_dim

        self.q_proj = nn.Linear(hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        batch = query.shape[0]
        q = self.q_proj(query).view(batch, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_2d(q, k, cos, sin, num_k_exclude_rope)

        # Training-time dropout is handled by the owning layer, so attention itself
        # is deterministic here; the checkpoint's dropout is inference noise.
        #
        # Fused scaled-dot-product attention instead of an explicit matmul + softmax:
        # the memory is the keys, and a full bank over a 64x64 trunk grid means a
        # query-by-key matrix of heads * 4096 * ~24k floats -- gigabytes of scores
        # materialised purely to be thrown away. The fused kernel never builds it.
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        out = out.transpose(1, 2).reshape(batch, -1, self.internal_dim)
        return self.o_proj(out)


class MemoryAttentionLayer(nn.Module):
    """One memory-attention block: self-attention -> cross-attention -> MLP."""

    def __init__(self, hidden_size: int = 256, num_attention_heads: int = 1, downsample_rate: int = 1,
                 feed_forward_hidden_size: int = 2048, dropout: float = 0.1, feed_forward_hidden_act: str = "relu"):
        super().__init__()
        self.self_attn = RoPEAttention(hidden_size, num_attention_heads, downsample_rate)
        # kv_in_dim=64: the bank is stored at the memory encoder's output width.
        self.cross_attn_image = RoPEAttention(hidden_size, num_attention_heads, downsample_rate, kv_in_dim=64)

        self.linear1 = nn.Linear(hidden_size, feed_forward_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_forward_hidden_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = ACTIVATIONS[feed_forward_hidden_act]()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, key_point_embedding: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        query = self.layer_norm1(queries)
        queries = queries + self.dropout1(self.self_attn(query, query, query, cos, sin))

        query = self.layer_norm2(queries)
        # Key carries the position code, value does not: the value is the memory
        # itself, untouched by positional information.
        query = self.cross_attn_image(
            query, keys + key_point_embedding, keys, cos, sin, num_k_exclude_rope
        )
        queries = queries + self.dropout2(query)

        query = self.layer_norm3(queries)
        query = self.linear2(self.dropout(self.activation(self.linear1(query))))
        return queries + self.dropout3(query)


class MemoryAttention(nn.Module):
    """Four blocks of memory attention over the bank plus the object pointers.

    Shape flow::

        features   (B, N, C)      C = 256
        memory     (B, M, 64)     the bank, at the memory encoder's width
        -> features (B, N, C)
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 4, num_attention_heads: int = 1,
                 downsample_rate: int = 1, feed_forward_hidden_size: int = 2048, dropout: float = 0.1,
                 feed_forward_hidden_act: str = "relu", rope_feat_sizes: tuple[int, int] = (64, 64),
                 rope_theta: float = 10000.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(hidden_size, num_attention_heads, downsample_rate,
                                 feed_forward_hidden_size, dropout, feed_forward_hidden_act)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.rotary_emb = VisionRotaryEmbedding(
            hidden_size, rope_feat_sizes, num_attention_heads, downsample_rate, rope_theta
        )
        #: Set by ``MemoryStack.set_gradient_checkpointing`` for training runs.
        self.gradient_checkpointing = False

    def forward(self, features: torch.Tensor, memory: torch.Tensor, features_pos: Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None, num_object_pointer_tokens: int = 0) -> torch.Tensor:
        output = features
        if features_pos is not None:
            # The 0.1 factor is the reference's; it keeps the absolute positional
            # code a small perturbation on top of the learned features.
            output = output + 0.1 * features_pos

        cos, sin = self.rotary_emb()
        # cos/sin are broadcast over the head and batch dims.
        cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
        sin = sin.view(1, 1, sin.shape[0], sin.shape[1])

        for layer in self.layers:
            output = run_block(
                layer, output, memory, memory_pos, cos, sin, num_object_pointer_tokens,
                recompute=self.gradient_checkpointing,
            )
        return self.layer_norm(output)


# ------------------------------------------------------------- memory encoder
class MemoryFuserBlock(nn.Module):
    """ConvNeXt-style block used to fuse the downsampled mask with image features."""

    def __init__(self, embed_dim: int = 256, intermediate_dim: int = 1024, kernel_size: int = 7,
                 padding: int = 3, layer_scale_init_value: float = 1e-6, hidden_act: str = "gelu"):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size,
                                        padding=padding, groups=embed_dim)
        self.layer_norm = ConvNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.activation = ACTIVATIONS[hidden_act]()
        self.pointwise_conv1 = nn.Linear(embed_dim, intermediate_dim)
        self.pointwise_conv2 = nn.Linear(intermediate_dim, embed_dim)
        self.scale = nn.Parameter(layer_scale_init_value * torch.ones(embed_dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(self.depthwise_conv(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.pointwise_conv2(self.activation(self.pointwise_conv1(hidden_states)))
        hidden_states = self.scale * hidden_states
        return residual + hidden_states.permute(0, 3, 1, 2)


class MemoryFuser(nn.Module):
    def __init__(self, num_layers: int = 2, embed_dim: int = 256, intermediate_dim: int = 1024,
                 kernel_size: int = 7, padding: int = 3, layer_scale_init_value: float = 1e-6,
                 hidden_act: str = "gelu"):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryFuserBlock(embed_dim, intermediate_dim, kernel_size, padding, layer_scale_init_value, hidden_act)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MaskDownSamplerLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2,
                 padding: int = 1, hidden_act: str = "gelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer_norm = ConvNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.activation = ACTIVATIONS[hidden_act]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer_norm(self.conv(x)))


class MaskDownSampler(nn.Module):
    """Downsample the predicted mask to the trunk's stride, widening channels as it goes."""

    def __init__(self, total_stride: int = 16, stride: int = 2, embed_dim: int = 256,
                 kernel_size: int = 3, padding: int = 1, hidden_act: str = "gelu"):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        if num_layers < 1:
            raise ValueError("total_stride must exceed stride")
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS[hidden_act]()
        in_chans, out_chans = 1, 1
        for _ in range(num_layers):
            out_chans = in_chans * (stride ** 2)
            self.layers.append(MaskDownSamplerLayer(in_chans, out_chans, kernel_size, stride, padding, hidden_act))
            in_chans = out_chans
        self.final_conv = nn.Conv2d(out_chans, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class MemoryEncoder(nn.Module):
    """Compress (predicted mask + unconditioned frame features) into one memory map.

    The mask is convolved down to the feature stride and **added element-wise** to
    the frame embedding, then fused with a couple of ConvNeXt blocks. No second
    image encoder is run -- the frame embedding is the one the encoder already
    produced, which is what keeps the design cheap.
    """

    def __init__(self, hidden_size: int = 256, output_channels: int = 64, mask_downsampler_embed_dim: int = 256,
                 mask_downsampler_kernel_size: int = 3, mask_downsampler_stride: int = 2,
                 mask_downsampler_padding: int = 1, mask_downsampler_total_stride: int = 16,
                 mask_downsampler_hidden_act: str = "gelu", fuser_num_layers: int = 2,
                 fuser_embed_dim: int = 256, fuser_intermediate_dim: int = 1024, fuser_kernel_size: int = 7,
                 fuser_padding: int = 3, fuser_layer_scale_init_value: float = 1e-6,
                 fuser_hidden_act: str = "gelu"):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(
            mask_downsampler_total_stride, mask_downsampler_stride, mask_downsampler_embed_dim,
            mask_downsampler_kernel_size, mask_downsampler_padding, mask_downsampler_hidden_act,
        )
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.memory_fuser = MemoryFuser(
            fuser_num_layers, fuser_embed_dim, fuser_intermediate_dim, fuser_kernel_size,
            fuser_padding, fuser_layer_scale_init_value, fuser_hidden_act,
        )
        self.position_encoding = SinePositionEmbedding(num_pos_feats=output_channels // 2, normalize=True)
        self.projection = nn.Conv2d(hidden_size, output_channels, kernel_size=1)
        #: Set by ``MemoryStack.set_gradient_checkpointing`` for training runs.
        self.gradient_checkpointing = False

    def forward(self, vision_features: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The mask arrives at full resolution, so the first downsampling layers run
        # on a 512x512x256 intermediate -- the single largest activation in the
        # stack. It is the first thing worth recomputing instead of storing.
        downsampled = run_block(self.mask_downsampler, masks, recompute=self.gradient_checkpointing)
        fused = self.feature_projection(vision_features) + downsampled
        fused = self.memory_fuser(fused)
        fused = self.projection(fused)
        position = self.position_encoding(fused.shape, fused.device, fused.dtype)
        return fused, position
