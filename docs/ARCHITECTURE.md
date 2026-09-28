# RotoStream architecture

How a click in the browser becomes a mask on every frame, and where each piece
lives. Line numbers refer to this commit; treat them as signposts, not a stable
API.

Three processes' worth of concern live here, but only two actual servers:

- **API** — FastAPI (`api/app`), one process, in-process background jobs.
- **Web** — Next.js App Router (`web/`), talks to the API over HTTP/JSON.
- **Model** — a plugin loaded inside the API process via the contract in
  `api/app/models/base.py`. See `docs/MODEL_CONTRACT.md`.

There is no broker, no database and no worker fleet. One editor, one GPU: the
exclusive gate in `api/app/jobs.py` serialises the expensive work, which is
simpler than Celery and enough for a studio tool.

---

## 1. Request lifecycle

```
upload ─► extract frames ─► prompt (preview) ─► propagate (job) ─► overlay ─► export
  │            │                   │                 │               │          │
videos.py   videos.py         tracking.py        pipeline.py      videos.py/exports.py
```

### 1.1 Upload — `POST /api/videos`

`api/app/routers/videos.py:106` (`upload_video`).

- Rejects the extension against `Settings.allowed_extension_set` (`settings.py:61`).
- Creates the video record and directories first:
  `Workspace.create_video` (`storage.py:57`) writes `meta.json` with
  `status="uploaded"`.
- Streams the body to `videos/<id>/source.<ext>` in 1 MiB chunks
  (`CHUNK_BYTES`, `videos.py:20`), enforcing `Settings.max_upload_bytes`
  (`settings.py:56`). An over-size upload deletes the whole video directory —
  no half-written records.
- Flips status to `extracting`, creates an `extract` job and submits it
  (`videos.py:150-152`). Returns `202` with the video and the `job_id`.

### 1.2 Extract frames — background job

`api/app/routers/videos.py:62` (`_extract_body`), running on a job thread.

1. `probe()` (`video.py:71`) shells out to `ffprobe`, records width/height/fps/
   duration/frame-count/audio in `meta.json`.
2. `target_frame_size()` (`video.py:116`) scales the long side down to
   `Settings.long_side` (default 960) keeping even dimensions.
3. `extract_frames()` (`video.py:132`) runs one `ffmpeg` pass to
   `frames/%06d.jpg`, capped at `Settings.max_frames` (default 900), using
   `_jpeg_qscale()` (`video.py:126`) to map quality 0-100 onto ffmpeg's qscale.
4. Marks the video `ready`, or records `status="failed"` with the error and
   re-raises so the job lands as `failed` too.

**Why pre-extract?** Masks are authored and exported at this working resolution,
so nothing downstream has to decode video per frame. The trade-off is baked into
the design: masks are `(H, W)` at this resolution, never at the encoder's internal
resolution.

### 1.3 Prompt — `POST /api/videos/{id}/preview`

`api/app/routers/tracking.py:54` (`preview`), the interactive click path.

- `assert_model_usable()` (`tracking.py:30`) fails fast with 404/409 if the model
  key is unknown or still a stub, instead of letting a background job die.
- `to_prompt_set()` (`prompts.py:17`) converts the JSON schema into the
  framework-free `PromptSet` the contract speaks.
- `preview_mask()` (`pipeline.py:115`) grabs the warm tracker (below), calls
  `reset()` then `add_prompt()`, and validates the result with
  `check_frame_result()`.
- The API returns the **rendered overlay PNG** plus `X-Mask-Area` /
  `X-Mask-Ratio` headers (`tracking.py:95-100`), so one click is one request and
  the browser does no per-pixel work. CORS exposes those headers
  (`main.py:85`).

### 1.4 Propagate — `POST /api/videos/{id}/track` (202 + job)

`api/app/routers/tracking.py:111` (`start_tracking`) validates prompts and creates
a session (`Workspace.create_session`, `storage.py:119`), then submits a `track`
job running `pipeline.run_tracking()` (`pipeline.py:155`).

Inside `run_tracking`:

1. `build_plan()` (`pipeline.py:47`) produces the leak-free step order (below).
2. `_instantiate()` (`pipeline.py:94`) creates and loads the tracker.
3. For each step: `add_prompt` or `propagate`, `check_frame_result`, then
   `save_mask()` (`masks.py:44`) to `sessions/<id>/masks/%06d.png` and a per-frame
   `{frame_index, score, object_present, prompted}` record.
4. `ctx.progress()` drives the SSE stream; `ctx.check_cancelled()` between frames
   makes cancel cooperative.
5. `tracker.memory_state()` is captured for the UI inspector, then the session
   `meta.json` is written (`Workspace.write_session_meta`, `storage.py:130`).

Progress is consumed by `GET /api/jobs/{id}/events`
(`tracking.py:182`), a Server-Sent Events stream that re-emits the job dict
whenever it changes and closes on a terminal status.

### 1.5 Overlay — `GET /api/videos/{id}/overlays/{index}`

`api/app/routers/videos.py:207` (`get_overlay`).

- Reads the working-resolution frame (`read_frame`, `video.py:166`) and the mask
  (`load_mask_or_empty`, `masks.py:25`), then renders RGBA server-side with
  `overlay_png_bytes()` (`masks.py:83`).
- When the request passes `session_id`, the response is cached immutably
  (`IMMUTABLE`, `videos.py:23`) — that is what makes scrubbing cheap, because the
  browser can keep every frame it has already fetched.

### 1.6 Export — `POST /api/videos/{id}/exports` (202 + job)

`api/app/routers/exports.py:68` (`create_export`) writes an export record and
submits an `export` job. `_export_body` (`exports.py:28`) builds `ExportInputs`
(`video.py:177`) and calls `run_export()` (`video.py:374`), which dispatches
through the `EXPORTERS` table (`video.py:355`):

| kind | encoder |
| --- | --- |
| `alpha_webm` | VP9 + `yuva420p` alpha |
| `overlay_mp4` | H.264 with the tinted mask |
| `replace_bg` | H.264, background blurred / black / white / green |
| `cutout_zip` | RGBA PNG sequence |
| `mask_zip` | binary mask PNGs |
| `mask_rle_json` | COCO-style RLE |

Every exporter composites to a temp PNG sequence and hands it to one `ffmpeg`
call, then removes the temp directory in a `finally`. Download is
`download_export` (`exports.py:126`).

---

## 2. On-disk layout

Defined by `api/app/storage.py` (`Workspace`, line 37); the module docstring is
the summary.

```
<workspace>/                              # Settings.resolved_workspace, default api/workspace
  videos/<video_id>/
    source.<ext>                          # original upload, never modified
    meta.json                             # VideoOut shape + probe/extract results
    frames/000000.jpg                     # working-resolution frames (masks match these)
    sessions/<session_id>/
      meta.json                           # SessionOut shape (scores, memory, prompt frames)
      masks/000000.png                    # binary masks, 0 / 255
    exports/
      files/<export_id>.<ext>             # artifacts (export_files_dir, storage.py:164)
      <export_id>.meta.json               # export record
```

Artifacts live one directory down (`exports/files/`) so that `*.meta.json`
records stay distinguishable from payloads that are *themselves* JSON — the RLE
export. That is the reason for the extra level; a flat `exports/` would make the
JSON export look like metadata.

`meta.json` writes go through a temp file + `replace()` (`storage.py:93`) so a
crash mid-write cannot leave a truncated record.

---

## 3. Jobs and the exclusive gate

`api/app/jobs.py`.

- `JobManager` (`jobs.py:92`) holds an ordered dict of jobs (bounded history) and
  one `threading.Semaphore(1)` (`jobs.py:98`).
- `submit(..., exclusive=True)` (`jobs.py:146`) starts a daemon thread per job.
  `_worker` (`jobs.py:154`) acquires the gate before running the body, so only one
  of tracking / export / extraction runs at a time — two GPU forwards at once
  would thrash VRAM, and ffmpeg already saturates cores.
- `JobContext` (`jobs.py:66`) is the only thing a job body gets: `progress()`,
  `note()` and `check_cancelled()`. Cancellation is **cooperative** — the body
  must call `check_cancelled()`, which raises `JobCancelled`; the worker maps that
  to `status="cancelled"`.
- Any other exception is caught, logged with a traceback, and turned into
  `status="failed"` plus the error string, so the UI always has something to show.

State is in-process and therefore lost on restart. That is a deliberate
consequence of "one editor, one GPU"; the on-disk session and export records are
the durable artifacts.

---

## 4. The warm-tracker preview cache

`api/app/pipeline.py:111-148`.

Loading model weights on every click would be unusable for a real network, so the
preview path keeps one warm tracker per `(video_id, model_key, checkpoint,
device)`, guarded by a single `threading.Lock` (a GPU cannot serve two overlapping
forwards). On each click it does:

```
tracker.reset()                 # clear the memory bank and prompts
result = tracker.add_prompt(prompt)   # re-seed from the new click
```

`reset()` + `add_prompt` is the contract's own restart path, so the preview never
needs a second instance and the interactive latency is one forward pass.
`clear_preview_cache()` (`pipeline.py:149`) exists for tests and for when a
checkpoint changes underneath the cache.

Tracking jobs do **not** use this cache: they build a fresh tracker per job so a
long run cannot be perturbed by a click that arrived mid-way.

---

## 5. The leak-free step planner

`api/app/pipeline.py:47` (`build_plan`).

Given prompts at frames `p1 < p2 < ... < pk`, it emits:

```
prompt(p1) -> propagate(p1+1 .. p2-1, FORWARD) -> prompt(p2) -> ... -> propagate(.., FORWARD to the end)
                                                                    \-> propagate(p1-1 .. 0, BACKWARD)
```

Each `propagate` therefore conditions only on frames already visited *in its own
direction*, which is exactly contract rule 4. The backward sweep is only appended
when the request asks for `bidirectional`, and it still never reads past the
prompt. This planner is the reason the causality tests can be strict: causality is
enforced by construction, not by hoping the tracker behaves.

`preview_mask` is the one place that clamps out-of-range prompt indices
(`_clamp`, `pipeline.py:42`); the tracking path clamps per prompt as well.

---

## 6. Where the pieces live (map)

| Concern | File |
| --- | --- |
| App wiring, CORS, model-registration bridge | `api/app/main.py` |
| Env config (all `ROTOSTREAM_*`) | `api/app/settings.py` |
| Disk layout | `api/app/storage.py` |
| ffprobe / extract / composite / encode | `api/app/video.py` |
| Mask IO, overlays, RLE | `api/app/masks.py` |
| Jobs, progress, cancellation, gate | `api/app/jobs.py` |
| Step planner, preview cache, run loop | `api/app/pipeline.py` |
| HTTP schemas | `api/app/schemas.py` |
| Routers | `api/app/routers/{health,models,videos,tracking,exports}.py` |
| The contract | `api/app/models/base.py` |
| Frame containers | `api/app/models/frames.py` |
| Plugin registry | `api/app/models/registry.py` |
| Reference baseline | `api/app/models/naive.py` |
| The model | `api/app/models/sam2_memory.py` |
| Eval / training harness | `ml/rotostream_ml/` |
| Frontend | `web/src/` |
