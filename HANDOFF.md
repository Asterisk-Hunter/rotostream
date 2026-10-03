# RotoStream handoff

Updated 2026-10-03. Start with [README](README.md), [results](docs/RESULTS.md),
[architecture](docs/ARCHITECTURE.md) and [deployment](docs/DEPLOYMENT.md).

## Current state

Upload, foreground/background prompts, causal tracking, saved immutable sessions,
inspection, cancellation and six exports are implemented. The color baseline
requires no weights. Neural inference independently implements SAM 2's prompt and
memory encoders, bank, attention, decoder and presence head over a reused frozen
Hiera encoder and released SAM 2.1 weights. This is a reproduction.

| Evidence | Result |
| --- | --- |
| DAVIS 2017 validation, first-frame masks, independent objects | 89.40 J&F; 30 clips, 61 objects, 3,862 scored object-frames |
| Color baseline, same quality protocol | 13.84 J&F |
| Neural propagation, float32, RTX 4050 Laptop GPU | 2.88 FPS, excluding initialization, prompts and scoring |
| Module parity | All checks within 1e-5; memory attention maximum difference 7.153e-7 |
| Python suite | 260 passed, 1 expected failure (including long-sweep memory regressions) |
| Frontend | 13 tests passed; lint, types and production build passed |
| Native HTTP smoke | 73 checks passed, including SSE and six exports |
| Training dry run | Three backward passes; finite loss, 272/305 trainable tensors with nonzero gradients |

[Raw reports](reports/benchmarks/) preserve the original measurement revision and
dirty working-tree flag. Synthetic ablations are regressions, not real-video
quality claims. No real-data fine-tuning or optimizer convergence is claimed.

## Start and verify

Requires Node 22.6+, pnpm 10.4.1, Python 3.12+, ffmpeg and ffprobe. Create a root
virtualenv, install API/ML requirements, then run frozen pnpm install and
`pnpm dev`. Optional neural dependencies are in `api/requirements-model.txt`;
install a compatible PyTorch/torchvision pair first. First use downloads weights.

```bash
pnpm verify
pnpm smoke
node scripts/python.mjs api/scripts/check_parity.py --json runs/parity.json
python scripts/render_results.py
```

Smoke requires a running API and creates then deletes a test clip. Use the
[included sample](samples/moving-square.mp4) for manual checks.

## Navigation

| Concern | Source |
| --- | --- |
| Contract and plugins | `api/app/models/base.py`, `registry.py` |
| Model and training | `api/app/models/sam2_memory.py`, `sam2_stack/` |
| Causal planner and preview cache | `api/app/pipeline.py` |
| Bounded worker and compute gate | `api/app/jobs.py` |
| Storage and restart recovery | `api/app/storage.py`, `main.py` |
| Video/export lifecycle | `api/app/video.py`, `routers/` |
| Studio and progress recovery | `web/src/` |
| Protocols, datasets, metrics, training | `ml/rotostream_ml/` |
| Deployment/CI | `compose.yaml`, `deploy/Caddyfile`, `.github/workflows/ci.yml` |

## Invariants

- Masks are boolean at source resolution; absence requires an empty mask.
- Propagation conditions only on visited frames in its direction; two-direction
  tracking resets state between causal sweeps.
- Mask prompts are copied read-only and cannot mix with point/box prompts.
- Preview and jobs share a compute gate; tracking evicts warm preview weights.
- Sessions cannot be overwritten; exports require successful sessions.
- Metadata publishes atomically, paths are contained, and media stays behind authentication.
- Freeze Hiera only; decoder skip projections must retain gradients.
- Trained-stack checkpoints restore the released encoder and validate contributed keys.
- Scores retain protocol, precision, weight revision and independent-object caveats.

## Operating boundary and follow-up

One editor, one object per session, one API worker. Jobs are ephemeral; restart
marks interrupted durable records failed for retry. Cancellation is cooperative,
including ffmpeg deadlines. Back up workspace before upgrades. The container
image supplies the CPU baseline; native GPU and remote access are in the runbook.
Use the latest authenticated-container CI result as the release gate.

Useful extensions are joint multi-object arbitration, durable jobs, tenant
ownership, real-data fine-tuning and full reference-policy comparison. Inspect
the weakest DAVIS tracks before changing heuristics. The
[original evaluation plan](docs/DAVIS-EVALUATION-PLAN.md) remains historical context.
