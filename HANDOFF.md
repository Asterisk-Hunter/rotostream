# RotoStream handoff

Updated 2026-10-06. Start with [README](README.md), [results](docs/RESULTS.md),
[architecture](docs/ARCHITECTURE.md), [deployment](docs/DEPLOYMENT.md) and the
[product performance report](docs/PRODUCT-PERFORMANCE.md).

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
| Python suite | 273 passed, 1 expected failure (including long-sweep memory regressions) |
| Frontend | 28 tests passed; lint, types and production build passed |
| Production dependency audit | No known vulnerabilities after pinning `source-map-js` 1.2.2 |
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

## Review states and deliverables (2026-10-06)

The studio reports what a run produced instead of implying that it finished.

- Review rules live in `api/app/quality.py`: a frame the tracker lost, a frame with
a mask but a score below `LOW_CONFIDENCE_THRESHOLD`, and a frame the user marked as
background stay three different answers. `web/src/lib/review.ts` mirrors them.
- `SessionOut` fills `coverage`, `sound`, `low_confidence_frames` and
`background_only_frames` from those rules, deriving them from a session's own scores
when the run predates the summary. The 480-frame baseline run with 198 missing
frames reports coverage 0.5875 and sound false instead of the old defaults.
- The studio lists, labels and jumps to problem frames, warns before exporting a
clip with missing masks, and never renders a "mask ready" claim.
- Every export records a manifest (container, codec, dimensions, fps, frames, audio,
options, mask-source coverage, warnings) built from the same facts the renderer uses;
`web/src/lib/exportPlan.ts` shows the same plan before rendering.
- Video exports mux the source audio when the upload has any (AAC in MP4, Opus in WebM).
- Sessions store the prompts that produced them, so a reload restores them and the
studio can say "prompts changed since this run".

The public `/docs` field guide is linked from the sign-in page and editor Help. It
explains the workflow, prompt terms, tracker trade-offs, review cues, export
formats, hosted limits, recovery steps and clip deletion. Timeline state
classification and review bins are memoized so moving through frames does not
rescan the complete score arrays;
review playback requests up to eight frames per second while the playhead advances
in clip time. Video exports now stream composed frames to ffmpeg instead of writing
and rereading temporary PNGs. H.264 exports use the `veryfast` preset, VP9 alpha
export uses `cpu-used 6`, and the lossless cutout PNG sequence uses compression
level 3. The local export suite passes for all six formats and verifies cutout
pixels exactly.

The profile mode in `api/scripts/smoke.py` can run a real clip through upload,
extraction, preview, full tracking via SSE, review assets and one or all exports;
it prints per-stage timings in a final `PROFILE` JSON line and deletes its test
upload by default. On 2026-10-06, an isolated local CPU run of a 10.01-second,
300-frame, 640×360 clip using `naive` completed the full workflow including all
six exports and downloads in 31.349 seconds. Tracking took 4.674 seconds;
sequential exports took 22.205 seconds. This is one local run, not a percentile or
a neural quality/speed claim. A controlled local `replace_bg` comparison measured
2.612 seconds at H.264 `medium` and 2.460 seconds at `veryfast`. Detailed
methodology and the authenticated Vercel session-mode profile command are in
[the performance report](docs/PRODUCT-PERFORMANCE.md). Hosted Cloud Run
wall-clock performance has not yet been measured.
An additional 20-second local API profile on the same day tracked 480 extracted
frames in 8.352 seconds and rendered one replaced-background export in 27.379
seconds (37.702 seconds end to end); the API reported CUDA available but used
`naive`. Export was 73% of this run. The source container's `nb_frames` field
reported 534, but full decode and extraction both yielded 480 frames. This is a
single uncontrolled local observation; see the report for its limits.
The GitHub production dependency audit exposed a high-severity `source-map-js`
issue inherited through PostCSS; the workspace override pins the patched 1.2.2
release and the production dependency audit now passes.
The full development dependency audit currently reports `braces@3.0.3` through
`eslint-config-next`; the GitHub advisory lists no patched version yet
([GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)). It is
not in the production dependency graph (`pnpm audit --prod` passes). Recheck the
upstream package and full audit when a fix is published.

Evidence from the same day: the API/ML suite passed (272 passed, 1 expected
failure), web lint, TypeScript, behavior tests and production build passed, and two
real exports were re-rendered against the masks they claim — `replace_bg` blur of a
300-frame clip and of the 480-frame baseline, both within H.264 rounding of the
reviewed masks (mean |diff| 1.6/255) with the background changed about twice as much
as the subject. The 480-frame export carries the source audio. Masks from a single
prompt stayed on the subject across the clip (inside-minus-ring luminance +63.6 mean,
+33.4 min over sampled frames; consecutive-mask IoU 0.941 mean, 0.707 min).

## Navigation

| Concern | Source |
| --- | --- |
| Contract and plugins | `api/app/models/base.py`, `registry.py` |
| Model and training | `api/app/models/sam2_memory.py`, `sam2_stack/` |
| Causal planner and preview cache | `api/app/pipeline.py` |
| Review rules and session coverage | `api/app/quality.py` |
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
- A run with missing or weak frames is never described as complete, and an export's
  manifest is built from the same facts as its artifact, so the two cannot disagree.
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
