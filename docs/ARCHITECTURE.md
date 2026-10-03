# RotoStream architecture

The Next.js studio talks to one FastAPI process containing a bounded worker and
tracker plugins. Disk stores frames, immutable sessions and exports. The production
gateway authenticates UI, API, media and SSE under one origin.

## Request lifecycle

1. Upload streams a permitted extension into a new record under a size limit.
   Queue admission or upload failure cleans up incomplete input.
2. ffprobe validates the source; bounded ffmpeg commands extract JPEG frames at
   working resolution. Source pixels and extracted frame counts are limited.
3. Preview converts coordinates into the tracker contract, acquires the shared
   compute gate and uses a bounded warm model cache. Validate the boolean
   source-resolution mask before returning an overlay PNG.
4. Tracking creates a new session, evicts preview weights and runs a causal
   prompt/propagation plan. Save each mask with confidence and presence metadata.
   Only completed runs publish success; failure and cancellation are durable.
5. Inspection retrieves frames, overlays, confidence and memory diagnostics for
   the chosen session. Later sessions cannot rewrite earlier masks.
6. Export requires a successful session and publishes a contained artifact after
   completion. Partial files are removed on failure; active deletion returns a conflict.

Schemas live in `api/app/schemas.py`, routes in `routers/`, planning in
`pipeline.py`, storage in `storage.py`, and encoding in `video.py`.

## Compute and consistency

One executor worker, 16 pending jobs and 200 terminal history records are the
defaults. History trimming preserves active work. Preview and background jobs
share one compute gate. The cache defaults to one warm tracker and is cleared
before tracking loads its own model. Cancellation is cooperative between operations.
The neural bank releases non-prompted frames outside its recent history, retaining
the current frame for retries and prompted conditioning anchors. Ordinary retained
history therefore stays constant as the clip grows.

IDs and artifact basenames are validated before path construction. JSON uses
unique temporary files, atomic replacement and a metadata lock. Media uses private
caching. The latest-session pointer advances only after successful tracking.

Startup marks interrupted extraction, tracking and export records failed.
Historical results survive even though the job table is lost. Shutdown cancels
outstanding work and waits for the worker. Multiple API workers or shared-volume
replicas are unsupported because ownership and compute locks are local.

```text
workspace/videos/<video-id>/
  meta.json
  source.<extension>
  frames/000000.jpg ...
  sessions/<session-id>/
    meta.json
    masks/000000.png ...
  exports/
    <export-id>.meta.json
    files/<export-id>.<extension>
```

## Model boundary

`VideoObjectTracker` receives a `FrameSource`, point/box or mask `PromptSet`
and emits `FrameResult`. Masks are boolean at source resolution; absence requires
an empty mask. `TrainableTracker` adds trainable parameters and sequence losses.
Builtins, entry points and configuration register plugins; neural dependencies
load lazily.

The neural implementation reuses pretrained Hiera and SAM 2.1 weights and
implements the prompt/memory encoders, directional bank, attention, decoder and
presence head. Forward sees lower visited indices; backward sees higher visited
indices. Two-direction tracking resets state between sweeps. Mask prompting
supports evaluation without changing browser point/box interaction.

See [model contract](MODEL_CONTRACT.md) and [results](RESULTS.md) for freeze
boundaries, parity and measurement scope.

## Browser behavior

Requests are keyed by clip, model and session. Abortable previews and stale response
guards prevent an old request replacing a new selection. Submission locks prevent
duplicate jobs. SSE drives progress with one retryable polling fallback; terminal
states stop monitoring. Reloading an extracting clip resumes polling.

Coordinates map rendered positions to working pixels; touch, pointer and keyboard
share that mapping. Completed overlays remain tied to their session until another
run succeeds. Frontend tests cover coordinates, timecodes and progress recovery;
HTTP smoke covers the service lifecycle.

## Operations

Only the loopback gateway is exposed by Compose. API and web containers run
without root. Readiness checks ffmpeg, writable storage and selected-model
availability without loading weights. Caddy protects every path. Defaults include
512 MiB uploads, 900 frames, a 960px long side and 600-second ffmpeg deadlines.

One object per session and one editor per deployment are the supported scope.
Authentication is an access boundary, not tenant ownership. The
[runbook](DEPLOYMENT.md) defines GPU setup, remote access, backups and upgrades.
CI verifies the authenticated container workflow.
