# RotoStream: from memory mechanism to usable studio

## Problem and outcome

Video cutouts need consistent masks across motion, occlusion and changes in
appearance. A frame-by-frame color matcher provides a useful plumbing baseline,
but it cannot reliably maintain identity. RotoStream reproduces SAM 2's memory
mechanism and makes it usable through an interactive rotoscoping application.

The pretrained Hiera encoder and released SAM 2.1 weights are reused. The memory
encoder, bank, attention, mask decoder, presence head, API, UI and evaluation
harness are implemented in this repository. The goal is a defensible reproduction
and complete editing workflow, not a novel segmentation architecture.

On DAVIS 2017 validation, the tiny model measures **89.40 J&F**, compared with
**13.84** for the color baseline using the same first-frame mask protocol, across
30 clips and 61 independently tracked objects. This evaluation excludes first and
last frames and averages by object. It does not merge overlapping object masks
into a joint label map or claim an official leaderboard entry. The measured
neural propagation speed is 2.88 FPS on an RTX 4050 Laptop GPU, excluding setup
and scoring. [Exact evidence and caveats](RESULTS.md).

## Engineering decisions

**One contract across inference, UI and evaluation.** `VideoObjectTracker`
separates model internals from HTTP, storage and datasets. Contract checks establish
mask resolution, boolean dtype, presence consistency, reset behavior and causal
forward/backward propagation. `TrainableTracker` gives the training harness its
own explicit boundary.

**Directional memory by construction.** Forward propagation attends only to lower
frame indices; backward propagation only to higher indices. The planner orders
multiple prompt frames without silently granting access to unvisited footage.
The studio's two-direction option is two causal sweeps, not offline future
conditioning. Poisoned-frame checks test this independently of the bank's metadata.

**Bounded compute and durable artifacts.** One worker and a bounded queue prevent
unlimited thread creation. Preview and long-running jobs share a compute gate,
and tracking evicts warm preview weights before loading another model. Immutable
sessions protect exports from re-tracking races. Metadata is published atomically;
restart recovery marks interrupted durable records failed rather than leaving them
apparently running forever.

**A frontend that survives real interaction.** Job monitoring combines SSE with
retryable polling, ignores stale responses, and handles vanished jobs after a
restart. Clip/model switches invalidate previews and async results. Submission
locks prevent duplicate jobs. Coordinate conversion is tested across rendered
sizes; touch and keyboard input support the same foreground/background prompts.

**A deployable single-editor service.** Non-root containers sit behind one
authenticated origin, including media and SSE. The API and web ports remain
private. Upload, source resolution, queue, cache and ffmpeg limits constrain the
workload. Readiness checks avoid model downloads. The runbook defines TLS access,
backups, restore and the one-worker deployment constraint.

## Bugs uncovered by verification

- The DAVIS loader selected the 2016 validation list when the archive contained
  both years. Strict 2017 selection and coverage checks prevent a plausible but
  incomplete result.
- One-click prompting was mislabeled as semi-supervised mask evaluation. Prompt
  mode, scoring window and aggregation are now explicit report metadata.
- Multimask pointer-token selection and pointer splitting used the wrong ordering.
  Decoder/presence/pointer parity now checks those paths, including raw logits
  before absence gating could hide differences.
- A trained-stack checkpoint omitted the frozen encoder, but inference could load
  a randomly initialized tower. Checkpoint loading now restores the released
  encoder and validates contributed keys. From-scratch initialization also
  preserves pretrained Hiera weights.
- The encoder freeze boundary also blocked decoder skip-projection gradients.
  The freeze now covers only Hiera; regression tests establish that decoder
  projections receive gradients and encoder parameters do not.
- Attention selected a bounded history, but the bank retained old frame tensors.
  Storage now releases unused non-prompted memory while keeping conditioning
  anchors and current-frame retry history. Long sweeps match an unpruned bank.

## Evidence and practical limits

The repository includes raw evaluator JSON/CSV, module parity evidence, gradient
wiring output, tests, a reproducible sample clip, and a real HTTP smoke workflow
through upload, tracking, SSE and six export formats. CI builds the actual container
stack and checks authentication as well as lint, types, frontend behavior and
Python contracts. [Deployment](DEPLOYMENT.md) · [results](RESULTS.md).

The studio handles one object per session and one editor per deployment. Background
jobs live in process memory and must be retried after a restart. In-flight ffmpeg
cancellation waits for the current command or its deadline. Real-data fine-tuning,
joint multi-object arbitration, multi-user tenancy and a full reference-tracker
policy comparison are outside the measured claims.

## Resume wording grounded in the repository

> Built a video rotoscoping studio with Next.js and FastAPI, implementing SAM 2's
> memory encoder, directional bank, attention and decoder over a frozen pretrained
> Hiera encoder; measured 89.40 J&F on all 30 DAVIS 2017 validation clips with
> independently tracked objects and first-frame mask prompts.

> Engineered bounded inference jobs, immutable tracking sessions, SSE progress with
> polling recovery, six export formats, and authenticated container deployment;
> verified model contracts, checkpoint safety and numerical module parity.

Do not replace “reproduction” with a novel-model claim, quote synthetic scores as
real-video quality, describe the mask-prompt result as one-click performance, or
claim real-time tracking from the measured throughput.
