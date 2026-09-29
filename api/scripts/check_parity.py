#!/usr/bin/env python
"""Numerical parity of the from-scratch memory modules against the released SAM 2.1 tree.

    python api/scripts/check_parity.py                     # default backbone
    python api/scripts/check_parity.py --backbone facebook/sam2.1-hiera-small

Why this exists
---------------
``docs/MODEL_CONTRACT.md`` claims the memory stack is a reimplementation of the
same architecture the released weights were trained with, and that those weights
load into it without a single missing or unexpected key. That claim is worth
nothing unless it is checked numerically, so this script compares the two
implementations on the same inputs:

* **memory encoder** -- mask + image embedding in, memory map + position codes out.
  Compared at 64x64, the resolution the trunk actually produces.
* **memory attention** -- the bank, the object pointers and the current frame's
  features in, conditioned features out.

Both are expected to agree to floating-point reassociation (~1e-6 in float32).
A larger difference means a real bug; the script exits non-zero in that case.

A note on the sizes
-------------------
The attention sequence length is set by the RoPE grid, and the reference's
table is precomputed for the full 64x64 trunk: comparing it as-is means
materialising a head x 4096 x 24k score matrix, i.e. gigabytes. The grid is
therefore overridden to 8x8 before either model is built. That is safe because
the cos/sin tables are non-persistent buffers -- they are not in any state_dict,
so they cannot affect weight loading -- and it exercises the same code path,
including the key-side RoPE repeat when the bank holds more than one frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import torch  # noqa: E402
from transformers import Sam2VideoConfig, Sam2VideoModel  # noqa: E402

# Same seed for both sides: the point is a deterministic input, not a lucky one.
SEED = 0
#: Attention tokens per side, chosen to match the reduced RoPE grid.
ROPE_GRID = (8, 8)
#: Above this, the difference is a bug rather than float32 reassociation.
TOLERANCE = 1e-5

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def compare(label: str, reference: torch.Tensor, built: torch.Tensor, tolerance: float) -> bool:
    """Print one comparison and return whether it is within tolerance."""
    if reference.shape != built.shape:
        print(f"  {RED}FAIL{RESET}  {label}: shape {tuple(reference.shape)} vs {tuple(built.shape)}")
        return False
    difference = (reference - built).abs().max().item()
    ok = difference <= tolerance
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {tag}  {label}: max|diff| {difference:.3e}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="facebook/sam2.1-hiera-tiny", help="HF id to compare against")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    from app.models.sam2_stack import MemoryStack

    torch.manual_seed(SEED)
    # The number of threads changes the reduction order, so a verbose run with an
    # unbounded pool reports differences that are scheduling, not maths.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    config = Sam2VideoConfig.from_pretrained(args.backbone)
    config.memory_attention_rope_feat_sizes = list(ROPE_GRID)
    reference = Sam2VideoModel.from_pretrained(args.backbone, config=config).eval()
    built = MemoryStack(config).eval()

    print(f"backbone: {args.backbone}")
    print(f"{DIM}the reference is sequence-first, the built stack is batch-first{RESET}\n")

    print("weight load")
    missing, unexpected = built.load_state_dict(reference.state_dict(), strict=False)
    keys_ok = not missing and not unexpected
    print(
        f"  {GREEN if keys_ok else RED}{'PASS' if keys_ok else 'FAIL'}{RESET}  "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )
    for name in missing[:5]:
        print(f"        missing: {name}")
    for name in unexpected[:5]:
        print(f"        unexpected: {name}")

    # ---------------------------------------------------------------- memory encoder
    print("\nmemory encoder")
    mask = torch.randn(1, 1, 1024, 1024)
    pixel_features = torch.randn(1, reference.hidden_dim, 64, 64)
    with torch.no_grad():
        ref_features, ref_positions = reference.memory_encoder(pixel_features, mask)
        got_features, got_positions = built.memory_encoder(pixel_features, mask)
    encoder_ok = compare("features", ref_features, got_features, args.tolerance)
    encoder_ok &= compare("position codes", ref_positions, got_positions, args.tolerance)

    # -------------------------------------------------------------- memory attention
    # Two banked frames plus four object-pointer tokens: the pointers exercise the
    # key-side RoPE exclusion, the second frame the repeat of the spatial table.
    print("\nmemory attention")
    grid = ROPE_GRID[0] * ROPE_GRID[1]
    pointer_tokens = 4
    banked = 2
    channels = config.memory_encoder_output_channels
    features = torch.randn(grid, 1, reference.hidden_dim)
    features_positions = torch.randn(grid, 1, reference.hidden_dim)
    memory = torch.randn(banked * grid + pointer_tokens, 1, channels)
    memory_positions = torch.randn(banked * grid + pointer_tokens, 1, channels)

    with torch.no_grad():
        ref_out = reference.memory_attention(
            current_vision_features=features,
            current_vision_position_embeddings=features_positions,
            memory=memory,
            memory_posision_embeddings=memory_positions,
            num_object_pointer_tokens=pointer_tokens,
        )
        got_out = built.memory_attention(
            features=features.transpose(0, 1),
            features_pos=features_positions.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_pos=memory_positions.transpose(0, 1),
            num_object_pointer_tokens=pointer_tokens,
        )
    # The reference returns (point_batch, batch, seq, ch); the built stack (batch, seq, ch).
    attention_ok = compare(
        "conditioned features",
        ref_out.squeeze(0).transpose(0, 1),
        got_out.transpose(0, 1),
        args.tolerance,
    )

    if keys_ok and encoder_ok and attention_ok:
        print(f"\n{GREEN}Parity holds.{RESET} The weights are the same weights.")
        return 0
    print(f"\n{RED}Parity FAILED.{RESET} See the numbers above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
