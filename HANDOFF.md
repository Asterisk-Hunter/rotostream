# HANDOFF

**Read this first.** Written for an AI agent or human picking this repo up cold.
Last updated at the end of the scaffolding session that built the API.

---

## 1. What this is

**RotoStream** — an AI rotoscoping studio. The user clicks an object once in a video
and the tracker propagates a mask across every frame, then exports an alpha matte /
transparent video / replaced background.

The interesting part is deliberately *not* finished: the user is implementing the
**SAM 2 memory-attention tracker from scratch themselves** and does not want help
with the model internals. Everything around it — contract, API, storage, ffmpeg,
jobs, UI, tests, evaluation — is this repo's job.

> **The one rule: do not implement the model.** Do not write `_memory_attention`,
> the memory encoder, the mask decoder or the occlusion head. The user owns
> `api/app/models/sam2_memory.py`. Everything else is fair game.

---

## 2. Status

| Area | State |
| --- | --- |
| Repo scaffold, git, venv, `.gitattributes` | **done** |
| Model plugin contract (`api/app/models/base.py`) | **done** |
| Plugin registry (builtin + entry points + env) | **done** |
| Reference baseline tracker (`naive.py`) | **done** |
| User's model stub (`sam2_memory.py`) | **done** (raises `NotImplementedError` by design) |
| API core: settings, storage, masks, ffmpeg, jobs, pipeline | **done** |
| HTTP routers (health, models, videos, tracking, exports) | **done** |
| Test suite (57 tests) + `check_model.py` | **done** — all green |
| **Next.js frontend (`web/`)** | **NOT STARTED** |
| **ML harness (`ml/`: DAVIS, J&F, train, evaluate, synthetic)** | **NOT STARTED** |
| Docs (`README.md`, `docs/MODEL_CONTRACT.md`, `docs/ARCHITECTURE.md`) | **NOT STARTED** |

Verified at handoff: `56 passed, 1 xfailed` on `api/tests`, and
`check_model.py` reports `8 passed, 0 failed` for `naive`.

---

## 3. Repo layout

```
rotostream/
  HANDOFF.md              <- you are here
  package.json            root scripts (dev launcher)
  scripts/dev.mjs         runs API + web together, prefixed logs, one Ctrl-C
  pytest.ini              testpaths = api/tests ml/tests
  .env.example            every env var, ROTOSTREAM_ prefixed
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
        sam2_memory.py    *** THE USER'S MODEL — stub, do not implement ***
      routers/            health, models, videos, tracking, exports
    scripts/check_model.py   standalone contract verification tool
    tests/                contract, naive, video, api + synthetic.py fixtures
  ml/                     training/eval harness (NOT STARTED)
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
   enforced by `test_no_future_leakage_forward` / `_backward` and by
   `check_model.py`.
4. Masks are `(H, W) bool` at **`FrameSource` resolution** (working resolution),
   not at the encoder's internal resolution.
5. `object_present=False` requires an all-`False` mask.

The step planner in `pipeline.py::build_plan` orders prompts and propagations so
rule 3 holds across multiple prompts: `prompt(p1) → propagate to p2-1 → prompt(p2)
→ … → propagate to the end → propagate backwards from p1-1 to 0`.

---

## 5. Run it

```bash
# one-time (already done in this checkout, but for a fresh clone):
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r api/requirements.txt

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

```bash
.venv/Scripts/python -m pytest api/tests           # 56 passed, 1 xfailed
.venv/Scripts/python api/scripts/check_model.py    # contract check, per tracker
```

`check_model.py <key>` is the fastest loop while iterating on a model — it prints
one line per check and pinpoints leakage.

---

## 7. Environment notes and sharp edges

- **Windows, git-bash.** Bash on Windows: use `ls`/`rm`/`mv`, forward slashes.
- **The venv is `--system-site-packages`.** `torch` (2.8.0+cu128), `opencv` and
  `scipy` come from the *global* Python 3.13 install, which is intentional so a
  2.5 GB torch download is avoided. `api/requirements.txt` uses loose `>=` bounds
  for the same reason — do not tighten them to exact pins.
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

---

## 8. Commit conventions

Per the repository owner, **explicitly requested**:

- **Never add a `Co-Authored-By: Codebuff` line.**
- **Never add a "Generated with Buffy" / "Generated with Codebuff" footer.**
- Commit messages are plain and descriptive: a subject line plus a body that
  explains *why*. No attribution, no emoji, no trailers.

---

## 9. Remaining work, in priority order

### 9.1 Frontend — `web/` (Next.js latest + TypeScript + Tailwind)

Not started. Suggested plan:

- Scaffold with
  `pnpm dlx create-next-app@latest web --ts --tailwind --eslint --app --src-dir --import-alias "@/*" --use-pnpm --disable-git --empty`
  (the `create-next-app` flags were verified against the installed version).
- `NEXT_PUBLIC_API_BASE_URL` (see `.env.example`) points at the API.
- Screens: video dropzone + project list → studio. In the studio: frame canvas with
  click-to-prompt (left click = positive, alt/shift = negative), mask overlay,
  timeline scrubber, mask-ratio/score sparkline, model picker from `GET /api/models`,
  export panel, memory-bank inspector from `SessionOut.memory`.
- Endpoints to wire (all under `/api`, see `http://127.0.0.1:8000/docs`):
  `POST /videos` (multipart) → poll the returned `job_id`; `GET /videos/{id}/frames/{i}`;
  `POST /videos/{id}/preview` (returns an **overlay PNG**, stats in `X-Mask-Area` /
  `X-Mask-Ratio` headers — CORS already exposes them); `POST /videos/{id}/track` →
  `GET /jobs/{id}/events` for SSE progress; `GET /videos/{id}/overlays/{i}`;
  `POST /videos/{id}/exports` → `GET /videos/{id}/exports/{eid}/download`.
- Overlays are rendered server-side as RGBA PNGs, so the canvas can just stack an
  `<img>` — no per-pixel client work needed.
- Pass `session_id` explicitly on overlay requests to get `immutable` caching, which
  is what makes scrubbing cheap.

### 9.2 ML harness — `ml/`

Not started. The user wants a defensible J&F number on DAVIS, so build:

- `ml/rotostream_ml/data/davis.py` — DAVIS 2017 download/loader returning sequences
  whose shapes match `TrainingSequence` (`frames (T,H,W,3) uint8`,
  `gt_masks (T,H,W) bool`, `prompts`).
- `ml/rotostream_ml/data/synthetic.py` — richer sibling of
  `api/tests/synthetic.py`: motion, occlusion, re-entry, with ground truth. The user
  explicitly asked for toy sequences to debug the memory module before real footage.
- `ml/rotostream_ml/metrics/jf.py` — region similarity **J**, boundary **F**, and
  **J&F**, matching the DAVIS protocol.
- `ml/rotostream_ml/evaluate.py` — CLI: `--tracker sam2_memory --dataset
  {synthetic,davis} --split val`, runs the plugin via the same registry, prints J/F/J&F.
  Must go through the contract, not a bespoke inference path.
- `ml/rotostream_ml/train.py` — harness that drives `TrainableTracker
  .trainable_parameters()` and `.training_step(TrainingSequence)`, owning the
  optimizer, schedule, checkpointing and logging. It must **not** reach into the
  user's model internals.
- `ml/tests/test_jf.py` — verify J/F against hand-computed tensors (the metric is the
  thing most likely to be silently wrong, and it is the number the user will quote).

Duplicate-ish note: `api/tests/synthetic.py` and `ml/.../synthetic.py` intentionally
overlap a little. Keeping the API test fixtures independent is worth the small
duplication — do not make `api/` depend on `ml/`.

### 9.3 Docs

- `README.md` — must frame **reused vs built** honestly, in the user's own terms:
  frozen pretrained image encoder reused; memory encoder, memory bank, memory
  attention, mask decoder and occlusion head built from scratch. Report the J&F
  numbers once `ml/` exists.
- `docs/MODEL_CONTRACT.md` — expand §4 above; the model author's single reference.
- `docs/ARCHITECTURE.md` — request lifecycle: upload → extract → prompt → track →
  overlay → export, and where each piece lives.

### 9.4 Smaller gaps worth closing

- No sample media in the repo. A tiny committed clip would make the frontend
  developable without hunting for a video.
- No CI. A `pytest` + `tsc --noEmit` workflow would be cheap and valuable.
- `SessionOut.scores` returns every frame; a 900-frame session is a large payload.
  Consider a `?include_scores=false` default with a separate scores endpoint.

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
