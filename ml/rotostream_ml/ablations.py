"""Run the inexpensive synthetic component regressions, never a capability benchmark.

    python -m rotostream_ml.ablations --device cuda --json reports/benchmarks/synthetic_ablations.json

Released weights are unchanged across rows. Each row resets video state before
every clip. This deliberately preserves directional memory; two sweeps are not
an experiment granting access to future conditioning frames.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path

from .evaluate import evaluate_tracker
from .synthetic import available, build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32",
                        help="bfloat16 autocast reduces CUDA activation memory; recorded with every row")
    parser.add_argument("--sequences", nargs="+", choices=available(), default=available())
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    import torch
    if args.precision == "bfloat16" and not args.device.startswith("cuda"):
        parser.error("--precision bfloat16 requires --device cuda")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    sequences = [build(name) for name in args.sequences]
    rows = [(f"memory_{count}", {"memory_bank_size": count}) for count in (1, 2, 4, 6, 8)]
    rows.extend((name, {"memory_bank_size": 6, flag: False}) for name, flag in (
        ("presence_head_off", "presence_head"),
        ("object_pointers_off", "object_pointers"),
        ("temporal_position_encoding_off", "temporal_position_encoding"),
    ))
    payload = {"created_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "synthetic_regression_only",
               "warning": "Author-created toy clips do not establish real-video capability or a DAVIS score.",
               "backbone": args.backbone, "device": args.device,
               "precision": args.precision,
               "memory_budget_definition": "recent spatial memories plus the prompted frame; offsets beyond the released table saturate",
               "runs": []}
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, settings in rows:
        print(f"\nAblation: {name}", flush=True)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bfloat16" else nullcontext()):
            result = evaluate_tracker("sam2_memory", sequences, device=args.device, dataset="synthetic",
                                      model_options={"backbone": args.backbone, **settings}, reuse_tracker=True)
        result.config["precision"] = args.precision
        payload["runs"].append({"name": name, "settings": settings, "evaluation": result.to_dict()})
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(result.report(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
