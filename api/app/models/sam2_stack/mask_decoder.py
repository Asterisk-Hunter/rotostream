"""The SAM-style two-way mask decoder.

Two attention directions alternate: prompt tokens attend to the image, then the
image attends back to the prompt tokens. The decoder also emits two extra signals
alongside the masks -- a predicted IoU per mask, and an object-presence logit --
which is what makes occlusions explicit rather than something the caller has to
infer from an empty mask.

High-resolution detail enters through **skip connections**: the stride-4 and
stride-8 Hiera features are added into the upsampling path and never pass through
memory attention.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Attention, ConvNorm, FeedForward, run_block

__all__ = ["MaskDecoder", "MaskDecoderOutput", "TwoWayAttentionBlock", "TwoWayTransformer"]

#: Score written into a mask when the presence head says the object is gone.
NO_OBJ_SCORE = -1024.0


class TwoWayAttentionBlock(nn.Module):
    """self-attention -> token-to-image -> MLP -> image-to-token."""

    def __init__(self, hidden_size: int = 256, num_attention_heads: int = 8, mlp_dim: int = 2048,
                 num_layers: int = 2, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_attention_heads, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.cross_attn_token_to_image = Attention(hidden_size, num_attention_heads, attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_dim, hidden_size, num_layers)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.layer_norm4 = nn.LayerNorm(hidden_size)
        self.cross_attn_image_to_token = Attention(hidden_size, num_attention_heads, attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, query_point_embedding: torch.Tensor,
                key_point_embedding: torch.Tensor,
                attention_similarity: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            # No residual here, deliberately: on the first layer the positional code
            # is skipped, so the attention output *replaces* the queries rather than
            # being added to them. Adding a residual is a silent, sizeable error.
            queries = self.self_attn(queries, queries, queries)
        else:
            query = queries + query_point_embedding
            queries = queries + self.self_attn(query, query, queries)
        queries = self.layer_norm1(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        queries = queries + self.cross_attn_token_to_image(query, key, keys, attention_similarity)
        queries = self.layer_norm2(queries)

        queries = self.layer_norm3(queries + self.mlp(queries))

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        keys = keys + self.cross_attn_image_to_token(key, query, queries)
        keys = self.layer_norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, hidden_size: int = 256, num_hidden_layers: int = 2, num_attention_heads: int = 8,
                 mlp_dim: int = 2048, attention_downsample_rate: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(hidden_size, num_attention_heads, mlp_dim, num_hidden_layers,
                                 attention_downsample_rate, skip_first_layer_pe=(i == 0))
            for i in range(num_hidden_layers)
        ])
        self.final_attn_token_to_image = Attention(hidden_size, num_attention_heads, attention_downsample_rate)
        self.layer_norm_final_attn = nn.LayerNorm(hidden_size)
        #: Set by ``MemoryStack.set_gradient_checkpointing`` for training runs.
        self.gradient_checkpointing = False

    def forward(self, point_embeddings: torch.Tensor, image_embeddings: torch.Tensor,
                image_positional_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = point_embeddings
        keys = image_embeddings
        for layer in self.layers:
            queries, keys = run_block(
                layer, queries, keys, point_embeddings, image_positional_embeddings,
                recompute=self.gradient_checkpointing,
            )

        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        queries = self.layer_norm_final_attn(queries + self.final_attn_token_to_image(query, key, keys))
        return queries, keys


class MaskDecoderOutput:
    """One frame's decoder output.

    ``low_res_masks`` and ``high_res_masks`` are ``(B, P, K, h, w)`` and
    ``(B, P, K, image_size, image_size)``; ``K`` is 3 for multi-mask output and 1
    otherwise. ``sam_tokens`` is the selected mask token per mask, which the stack
    projects into an object pointer.
    """

    __slots__ = ("low_res_masks", "high_res_masks", "iou_scores", "object_score_logits", "sam_tokens")

    def __init__(self, low_res_masks, high_res_masks, iou_scores, object_score_logits, sam_tokens):
        self.low_res_masks = low_res_masks
        self.high_res_masks = high_res_masks
        self.iou_scores = iou_scores
        self.object_score_logits = object_score_logits
        self.sam_tokens = sam_tokens


class MaskDecoder(nn.Module):
    """Predict masks, their quality, and whether the object is present at all."""

    def __init__(self, hidden_size: int = 256, num_multimask_outputs: int = 3, num_hidden_layers: int = 2,
                 num_attention_heads: int = 8, mlp_dim: int = 2048, attention_downsample_rate: int = 2,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256,
                 dynamic_multimask_via_stability: bool = True, dynamic_multimask_stability_delta: float = 0.05,
                 dynamic_multimask_stability_thresh: float = 0.98):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        # Token order is load-bearing: [presence, IoU, mask_0, mask_1, ...].
        self.iou_token = nn.Embedding(1, hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_size)
        self.obj_score_token = nn.Embedding(1, hidden_size)

        self.transformer = TwoWayTransformer(hidden_size, num_hidden_layers, num_attention_heads,
                                             mlp_dim, attention_downsample_rate)

        self.upscale_conv1 = nn.ConvTranspose2d(hidden_size, hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(hidden_size // 4, hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = ConvNorm(hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        self.output_hypernetworks_mlps = nn.ModuleList([
            FeedForward(hidden_size, hidden_size, hidden_size // 8, 3) for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = FeedForward(hidden_size, iou_head_hidden_dim, self.num_mask_tokens,
                                               iou_head_depth, sigmoid_output=True)
        # One feed-forward on the presence token -> a visibility logit.
        self.pred_obj_score_head = FeedForward(hidden_size, hidden_size, 1, 3)

        # Skip-connection projections for the stride-4 and stride-8 features.
        self.conv_s0 = nn.Conv2d(hidden_size, hidden_size // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    # ------------------------------------------------------------------ helpers
    def _stability_scores(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """How stable a mask is under a small logit threshold perturbation.

        A mask that barely changes between the two thresholds is one the decoder
        is confident about; a mask on the edge of falling apart is not.
        """
        flat = mask_logits.flatten(-2)  # (N, K, h*w)
        delta = self.dynamic_multimask_stability_delta
        area_i = (flat > delta).sum(dim=-1).float()
        area_u = (flat > -delta).sum(dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, torch.ones_like(area_i))

    def _dynamic_multimask(self, logits: torch.Tensor, iou: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fall back to the best multi-mask output when the single mask is unstable.

        Operates on ``(N, K, h, w)`` where ``N = batch * point_batch``.
        """
        multi_logits, multi_iou = logits[:, 1:], iou[:, 1:]
        best = torch.argmax(multi_iou, dim=-1)
        gather_idx = best[:, None, None, None].expand(-1, 1, multi_logits.size(-2), multi_logits.size(-1))
        best_logits = torch.gather(multi_logits, 1, gather_idx)
        best_iou = torch.gather(multi_iou, 1, best[:, None])

        single_logits, single_iou = logits[:, 0:1], iou[:, 0:1]
        stable = self._stability_scores(single_logits) >= self.dynamic_multimask_stability_thresh

        logits_out = torch.where(stable[..., None, None], single_logits, best_logits)
        iou_out = torch.where(stable, single_iou, best_iou)
        return logits_out, iou_out

    # ------------------------------------------------------------------ forward
    def forward(self, image_embeddings: torch.Tensor, image_positional_embeddings: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool, high_resolution_features: list[torch.Tensor],
                image_size: int) -> MaskDecoderOutput:
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        flat = batch_size * point_batch_size

        output_tokens = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)

        # The dense prompt is summed into the image features, not concatenated.
        image_embeddings = (image_embeddings + dense_prompt_embeddings).repeat_interleave(point_batch_size, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, dim=0)

        queries = tokens.reshape(flat, tokens.shape[2], num_channels)
        keys = image_embeddings.flatten(2).permute(0, 2, 1)
        key_positions = image_positional_embeddings.flatten(2).permute(0, 2, 1)
        queries, keys = self.transformer(queries, keys, key_positions)

        iou_token_out = queries[:, 1, :]
        mask_tokens_out = queries[:, 2 : 2 + self.num_mask_tokens, :]
        object_token_out = queries[:, 0, :]

        image_embeddings = keys.transpose(1, 2).view(flat, num_channels, height, width)

        # Skip connections: high-resolution encoder features bypass memory attention.
        feat_s0, feat_s1 = high_resolution_features
        feat_s0 = feat_s0.repeat_interleave(point_batch_size, dim=0)
        feat_s1 = feat_s1.repeat_interleave(point_batch_size, dim=0)

        upscaled = self.upscale_conv1(image_embeddings) + feat_s1
        upscaled = self.activation(self.upscale_layer_norm(upscaled))
        upscaled = self.activation(self.upscale_conv2(upscaled) + feat_s0)

        hyper_in = torch.stack(
            [mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)], dim=1
        )
        _, _, out_h, out_w = upscaled.shape
        masks = (hyper_in @ upscaled.view(flat, -1, out_h * out_w)).view(
            flat, self.num_mask_tokens, out_h, out_w
        )
        iou_pred = self.iou_prediction_head(iou_token_out)

        if multimask_output:
            masks, iou_pred = masks[:, 1:], iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask(masks, iou_pred)
        else:
            masks, iou_pred = masks[:, 0:1], iou_pred[:, 0:1]

        # The mask used to build memory is a *hard* choice: if the presence head
        # says the object is gone, the frame collapses to "no object" rather than
        # to a stale mask.
        present = self.pred_obj_score_head(object_token_out)
        is_present = present > 0
        masks = torch.where(is_present[:, :, None, None], masks, torch.full_like(masks, NO_OBJ_SCORE))

        high_res_masks = F.interpolate(
            masks.float(), size=(image_size, image_size), mode="bilinear", align_corners=False
        ).to(masks.dtype)

        best = torch.argmax(iou_pred, dim=-1)
        sam_tokens = mask_tokens_out[torch.arange(flat, device=masks.device), best]

        shape = lambda tensor: tensor.view(batch_size, point_batch_size, *tensor.shape[1:])  # noqa: E731
        return MaskDecoderOutput(
            low_res_masks=shape(masks),
            high_res_masks=shape(high_res_masks),
            iou_scores=shape(iou_pred),
            object_score_logits=present.view(batch_size, point_batch_size, 1),
            sam_tokens=shape(sam_tokens),
        )
