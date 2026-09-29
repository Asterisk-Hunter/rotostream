# HANDOFF

**Read this first.** Written for an AI agent or human picking this repo up cold.
Last updated after the session that implemented the memory stack and finished the
frontend, the ML harness and the docs.

---

## 1. What this is

**RotoStream** — an AI rotoscoping studio. The user clicks an object once in a video
and the tracker propagates a mask across every frame, then exports an alpha matte /
transparent video / replaced background.

The engineering claim is the split between **reused** and **built**, and it is worth
stating precisely because it is the first question anyone asks:

| Reused | Built here, from scratch |
| --- | --- |
| The Hiera image encoder (frozen, pretrained) | Memory encoder (mask → memory map) |
| The released SAM 2.1 weights, as a warm start | Bounded, directional memory bank |
| The `Sam2VideoConfig` that defines the architecture | Memory attention: 2D axial RoPE, object pointers |
| The tracker plugin contract | Two-way mask decoder + occlusion/presence head |
| | The training objective, the harness, the API, the UI |

The memory stack is **not** a wrapper around `Sam2VideoModel`. It is a separate
implementation that loads the released weights key-for-key and reproduces the
reference numbers bit-exactly — verified by `api/scripts/check_parity.py`, which is
the single most useful command in this repo (see §6).

Everything else — contract, API, storage, ffmpeg, jobs, UI, tests, evaluation — is
this repo's own work.

---

## 2. Status

| Area | State |
| --- | --- |
| Repo scaffold, git, venv, `.gitattributes` | **done** |
| Model plugin contract (`api/app/models/base.py`) | **done** |
| Plugin registry (builtin + entry points + env) | **done** |
| Reference baseline tracker (`naive.py`) | **done** |
| **Memory-attention tracker (`sam2_memory.py` + `sam2_stack/`)** | **done**, `implemented=True` |
| API core: settings, storage, masks, ffmpeg, jobs, pipeline | **done** |
| HTTP routers (health, models, videos, tracking, exports) | **done** |
| Test suite (`api/tests` + `ml/tests`) | **done** — 205 passed, 1 xfailed |
| Next.js frontend (`web/`) | **done** |
| ML harness (`ml/`: DAVIS, J&F, train, evaluate, synthetic, leakcheck) | **done** |
| Docs (`README.md`, `docs/MODEL_CONTRACT.md`, `docs/ARCHITECTURE.md`) | **done** |
| CI (`.github/workflows/ci.yml`: pytest + `tsc --noEmit`) | **done** |

Verified numbers from the last full pass are in §6, including the residual gaps
(no DAVIS J&F yet — see §9).

---

## 3. Repo layout

```
rotostream/
  HANDOFF.md              <- you are here
  README.md               reused-vs-built framing, quickstart, results
  docs/
    MODEL_CONTRACT.md     the model author's reference (contract + memory maths)
    ARCHITECTURE.md       request lifecycle, file:line pointers, storage layout
  package.json            root scripts (dev launcher)
  scripts/dev.mjs         runs API + web together, prefixed logs, one Ctrl-C
  pytest.ini              testpaths = api/tests ml/tests
  web/.env.example        NEXT_PUBLIC_API_BASE_URL for the browser
  .env.example            every server env var, ROTOSTREAM_ prefixed
  .github/workflows/ci.yml
  .venv/                  Python venv, --system-site-packages (see §7)
  api/
    app/
      main.py             FastAPI app, CORS, lifespan, model registration bridge
      settings.py         pydantic-settings, ROTOSTREAM_* env
      schemas.py          all request/response models
      storage.py          Workspace: disk layout for videos/sessions/exports
      video.py            ffprobe, frame extraction, compositing, export encoders
      masks.py            mask IO, RGBA overlay rendering, COCO RLE
      jobs.py             thread job manager: progress, cancel, exclusive gate
      pipeline.py         leak-free step planner + run_tracking + preview cache
      prompts.py          API schema  <->  contract PromptSet
      models/
        base.py           *** THE CONTRACT — read this file ***
        frames.py         FrameSource: ArrayFrameSource, DirectoryFrameSource
        registry.py       plugin discovery (builtin / entry points / env)
        naive.py          reference baseline (memory-free, on purpose)
        sam2_memory.py    the tracker: contract glue + inference + training step
        sam2_stack/       *** THE MODEL *** the from-scratch memory stack
          backbone.py       frozen Hiera wrapper, resize/normalise frames
          modules.py        prompt encoder, RoPE attention, memory encoder/attention
          mask_decoder.py   two-way transformer + mask/IoU/presence heads
          model.py          MemoryBank (directional gather) + MemoryStack assembly
          losses.py         focal + dice + IoU-L1 + presence CE
      routers/            health, models, videos, tracking, exports
    scripts/
      check_model.py      standalone contract verification tool
      check_parity.py     numerical parity vs the released SAM 2.1 modules
    tests/                contract, naive, video, api + synthetic.py fixtures
  ml/
    requirements.txt
    rotostream_ml/        davis, evaluate, leakcheck, metrics, sequences,
                          synthetic, train
    tests/                davis, evaluate, leakcheck, metrics, synthetic, train
  web/                    Next.js app: Studio, FrameStage, Timeline, Panels, Rail
```

---

## 4. The contract (the seam you must not break)

`api/app/models/base.py` is the only file a model author needs. Five methods:

```python
class VideoObjectTracker(ABC):
    key: str                                   # registry name
    @classmethod
    def info(cls) -> TrackerInfo               # description, uses_memory, trainable, implemented
    def load(self, *, device="auto", checkpoint=None) -> None
    def set_video(self, frames: FrameSource) -> None
    def add_prompt(self, prompts: PromptSet) -> FrameResult
    def propagate(self, frame_index: int, direction: Direction) -> FrameResult
    def reset(self) -> None                    # optional, default no-op
    def memory_state(self) -> dict             # optional, powers the UI inspector
```

Hard rules the app relies on:

1. `load` → `set_video` → `add_prompt` → repeated `propagate`. In that order.
2. `add_prompt` must **both** return the prompted frame's mask **and** store that
   frame into the memory bank.
3. **`propagate` must only attend to already-visited frames.** `FORWARD` may use
   indices `< frame_index`; `BACKWARD` may use `> frame_index`. Reading the
   unvisited side is future leakage and inflates benchmark numbers. This is
   enforced by `test_no_future_leakage_forward` / `_backward`,
   `check_model.py`, and independently by `rotostream_ml.leakcheck`.

   **This is a deliberate deviation from the paper**, which reads memory
   bidirectionally. Do not "fix" it to match the paper: the whole point of this
   repo is a tracker you cannot cheat with, and an offline bidirectional pass
   would use frames that an interactive session does not have yet.
4. Masks are `(H, W) bool` at **`FrameSource` resolution** (working resolution),
   not at the encoder's internal resolution.
5. `object_present=False` requires an all-`False` mask.

The step planner in `pipeline.py::build_plan` orders prompts and propagations so
rule 3 holds across multiple prompts: `prompt(p1) → propagate to p2-1 → prompt(p2)
→ … → propagate to the end → propagate backwards from p1-1 to 0`.

**Inference runs under `torch.no_grad()`** (`sam2_memory._inference` wraps
`add_prompt` / `propagate`). Only `training_step` builds a graph. This is not a
micro-optimisation: the bank stores tensors derived from decoder outputs, so an
un-detached graph is pinned by the bank and grows with the clip until the process
dies. See §7.

---

## 5. Run it

```bash
# one-time (already done in this checkout, but for a fresh clone):
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r api/requirements.txt -r ml/requirements.txt

# both servers, prefixed logs, Ctrl-C stops both
pnpm dev                       # or: node scripts/dev.mjs
node scripts/dev.mjs --api-only
node scripts/dev.mjs --web-only

# API alone
cd api && ../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
# -> http://127.0.0.1:8000/docs
```

`ROTOSTREAM_DEFAULT_MODEL` selects the tracker (`naive` by default).
`ROTOSTREAM_DEVICE=auto|cuda|cpu`. Full list in `.env.example`.

---

## 6. Verify it

Fast, and no model weights involved:

```bash
.venv/Scripts/python -m pytest api/tests ml/tests   # 205 passed, 1 xfailed
cd web && pnpm exec tsc --noEmit
```

Then, in increasing order of cost. **This is the intended reading order** — each
step fails differently, so a failure localises the problem.

```bash
# 1. Contract: shapes, dtypes, reset, leakage. ~2 min on CPU.
.venv/Scripts/python api/scripts/check_model.py sam2_memory

# 2. Numerical parity against the released SAM 2.1 modules. ~1 min on CPU.
.venv/Scripts/python api/scripts/check_parity.py

# 3. Causality, scenario by scenario. ~5 min on a GPU with --max-checks 3.
cd ml && ../.venv/Scripts/python -m rotostream_ml.leakcheck \
    --model sam2_memory --all --device cuda --max-checks 3

# 4. End-to-end quality on the synthetic benchmarks. ~1 min on a GPU.
cd ml && ../.venv/Scripts/python -m rotostream_ml.evaluate \
    --model sam2_memory --dataset synthetic --device cuda
```

Last recorded output:

| Check | Result |
| --- | --- |
| `pytest api/tests ml/tests` | 205 passed, 1 xfailed |
| `check_model.py sam2_memory` | 8 passed, 0 failed, 0 skipped |
| `check_parity.py` | 0 missing / 0 unexpected keys; encoder features, position codes and attention output all `max|diff| 0.000e+00` |
| `leakcheck --all --max-checks 3` | 5 scenarios × 2 directions, all PASS |
| `evaluate --dataset synthetic` | mean **J&F 0.96** over 5 sequences / 136 frames (color_shift, distractor, linear, reentry 1.0000; occlusion 0.8000) |
| `train --dry-run` | PASS — loss 0.0424, gradients through 305 tensors, 274 non-zero |

Training (needs a GPU; see §7 for the memory story):

```bash
cd ml
../.venv/Scripts/python -m rotostream_ml.train --model sam2_memory --dry-run --device cuda
../.venv/Scripts/python -m rotostream_ml.train --model sam2_memory --overfit --steps 50 --device cuda
../.venv/Scripts/python -m rotostream_ml.train --model sam2_memory --steps 200 \
    --checkpoint-dir runs/train --device cuda
```

The smoke test against a *running* server (uploads a clip, tracks, exports) is
`api/scripts/smoke.py`.

---

## 7. Environment notes and sharp edges

- **Windows, git-bash.** Bash on Windows: use `ls`/`rm`/`mv`, forward slashes.
- **The venv is `--system-site-packages`.** `torch` (2.8.0+cu128), `opencv` and
  `scipy` come from the *global* Python 3.13 install, which is intentional so a
  2.5 GB torch download is avoided. `api/requirements.txt` uses loose `>=` bounds
  for the same reason — do not tighten them to exact pins.
- **`transformers>=4.57` is required *only* by `sam2_memory`.** It supplies
  `Sam2VideoConfig` (which defines the architecture) and `Sam2VisionModel` (the
  frozen encoder). It is imported lazily inside the tracker, so the API, the naive
  tracker and every endpoint work without it.
- **ffmpeg 7.1.1 must be on PATH.** `/api/health` reports `ffmpeg: true|false`.
  Tests that need it are marked `@requires_ffmpeg` and skip cleanly.
- **Sharp edge: `python - <<'PY'` plus any multiprocessing/spawn worker recurses
  infinitely** on Windows and hangs (hit this during a benchmark). If you need
  workers, write a real `.py` file with an `if __name__ == "__main__":` guard.
- **`.gitattributes` forces LF.** Windows checkouts will not churn line endings.
- **`app.routes` is misleading on current FastAPI.** Routers appear as lazy
  `_IncludedRouter` objects that have no `.path`, so counting them looks like the
  routers never registered. Assert against `app.openapi()["paths"]` instead.
- **The registry reads `os.environ`, not the settings object.** Entries set in
  `.env` are bridged in `main.py::apply_model_registrations`. If you add config
  that the registry should see, bridge it there.
- **Sibling project.** `C:\Data\temp` is an unrelated masked-autoencoder project
  using the same global Python. Its `gradio` is pinned `<6` because the global
  env has `anyio 3.7.1`. Do not upgrade `anyio` globally without checking it.

### Sharp edges specific to the model

These all cost real debugging time. They are recorded because every one of them is
easy to reintroduce.

- **Shape mismatch: the reference is sequence-first, the built stack is
  batch-first.** `Sam2VideoMemoryAttention` takes `(seq, batch, ch)`; our
  `MemoryAttention` takes `(batch, seq, C)`. Feeding the wrong one does not raise
  a clean error — the reference's key-side RoPE repeat computes
  `repeat_factor = k_len // q_len`, so a wrong layout multiplies the cos/sin table
  into a 16 GB tensor and the machine dies. **This has already taken this PC down
  once.** Compare tensors with `check_parity.py`, which transposes explicitly.
- **Attention must not be materialised.** With a full bank (6 recent + 1 prompted
  frame, 4096 tokens each) an explicit `q @ k.T` is ~3 GB *per layer* in float32.
  `RoPEAttention` uses `F.scaled_dot_product_attention`, which is also what makes
  the output bit-exact against the reference.
- **Inference must not build a graph** (see §4). The bank keeps frame features,
  position codes and pointers alive, so an un-detached graph across a clip is a
  slow leak; the failure mode is a segfault, not a Python exception.
- **Training needs gradient checkpointing to fit on a consumer GPU.**
  `MemoryStack.set_gradient_checkpointing(True)`, enabled by
  `training_step`, recomputes activations in the backward pass. The activation
  hog is the memory encoder's stride-2 conv on a 1024×1024 mask (512×512×256 per
  frame). Without it, a 16-frame training step sat pinned at the 6 GB ceiling for
  well over ten minutes on an RTX 4050 and thrashed the machine; with it, a step
  is ~10-15 s and a 50-step overfit completes.
- **`trainable_parameters()` / `state_dict()` / `training_step()` must not require
  `set_video()`.** The harness asks for parameters immediately after `load()`. The
  readiness check is split into `_require_loaded()` (weights exist) and
  `_require_ready()` (weights + a video; inference only).
- **`training_step` returns only scalar loggable tensors.** The harness logs every
  value in the returned dict as a scalar, so the chosen-mask index stays out of
  `to_dict()` — averaging an index is meaningless and `mean()` on a Long tensor
  raises.

---

## 8. Commit conventions

Per the repository owner, **explicitly requested**:

- **Never add a `Co-Authored-By: Codebuff` line.**
- **Never add a "Generated with Buffy" / "Generated with Codebuff" footer.**
- Commit messages are plain and descriptive: a subject line plus a body that
  explains *why*. No attribution, no emoji, no trailers.

---

## 9. Remaining work, in priority order

### 9.1 The DAVIS number — the one real gap

`README.md` reports synthetic J&F only, because a DAVIS number is only meaningful
after a real training run and there is no DAVIS data in this checkout. To close it:

```bash
python -m rotostream_ml.evaluate --model sam2_memory --dataset davis \
    --root <path/to/DAVIS> --split val --device cuda --json runs/davis_val.json
```

Then paste the printed J&F into the README table (never by hand — copy the
command's output) and fill the ablation skeleton beside it with
`memory_bank_size` variants. Until then the README says *pending* on purpose.

### 9.2 Sample media

There is no clip in the repo, so the frontend needs a video before it shows
anything interesting. A tiny committed clip (a few hundred KB) would make the
demo reproducible for a reviewer.

### 9.3 `SessionOut.scores` payload

`scores` returns one entry per frame; a 900-frame session is a large payload.
The frontend's `Timeline` consumes it, so **do not change the default** without
updating `web/src/components/Timeline.tsx` in the same commit. The intended fix is
a `?include_scores=false` default plus a dedicated scores endpoint.

### 9.4 Mask-ratio sparkline

Deliberately not built. The API stores per-frame masks (fetchable as PNG) but
exposes no per-frame area series, so a sparkline would mean either N mask fetches
per session or a new aggregate field. `FrameResult.extras` is the natural place to
carry `mask_ratio` per frame if it is ever wanted — check whether anything reads
`extras` before changing its contents.

### 9.5 Multi-object sessions

One tracked object per session today; multiple objects would need per-object memory
banks. Multiple sessions per video cover the current brief.

---

## 10. Decisions already made, and why

- **Reference baseline is deliberately naive and memory-free.** It makes the app work
  today *and* it is the failure mode the memory stack must beat —
  `test_baseline_tracks_colour_not_identity` documents it: with no memory bank the
  tracker adopts a same-coloured impostor as the tracked object.
- **Single tracked object per session.** Multi-object would need per-object memory
  banks; the user's brief is one object. Multiple sessions per video cover the rest.
- **Frames are pre-extracted to JPEG at a working resolution** (long side 960 by
  default) rather than decoded from the video per request. Masks are authored and
  exported at that resolution — simple, cacheable, and honest.
- **In-process job manager, not Celery/Redis.** One editor, one GPU; the exclusive
  semaphore serialises heavy jobs. Revisit only if the API ever serves multiple users.
- **Preview keeps a warm tracker per (video, model, checkpoint)** with a lock, and
  calls `reset()` + `add_prompt` per click. Loading weights on every click would be
  unusable for a real model.
- **Line endings normalised to LF** so the Windows checkout does not churn.
- **The released weights are a warm start, not a crutch.** `MemoryStack.load_reference_weights`
  loads them because it is the only way to *prove* the architecture matches; the
  training harness then trains the memory stack on top. `reference_weights=False`
  builds a randomly initialised stack with the same shape, which is what the
  training entry point can use to prove the loop learns rather than remembers.
- **Numerical parity is a test, not a vibe.** `check_parity.py` is kept in the repo
  because "I reimplemented SAM 2's memory modules" is a claim that should be
  falsifiable in one command.
- **The directional-memory deviation stands.** See §4 rule 3. It is enforced by
  three independent checks and is the reason the project is about *leak-free*
  tracking rather than another mask quality number.
