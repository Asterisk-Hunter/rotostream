# Deploy and operate RotoStream

RotoStream is a **single-editor, single-process studio**. Video, session and export
records live on disk; jobs and model caches live in memory. The shipped deployment
protects the entire studio with a password, keeps backend ports private, and uses
the same browser origin for requests, images, downloads and SSE.

## Container deployment

Requires Docker Engine with Compose v2. From the repository root:

```bash
cp .env.production.example .env.production
docker run --rm -it caddy:2.10-alpine caddy hash-password
# Put the generated bcrypt hash in ROTOSTREAM_AUTH_HASH in .env.production.
# Keep the single quotes around it so Compose preserves the dollar signs.
docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production up --build -d --wait
```

PowerShell uses `Copy-Item .env.production.example .env.production` for the first
line. Open `http://127.0.0.1:8080` and sign in with the configured username and the
password you hashed. An unset hash prevents Compose from starting. An incorrect
hash fails gateway startup. `.env.production` is ignored by Git.

The default image includes the **CPU color baseline**, ffmpeg and ffprobe. It does
not install PyTorch or download model weights. It is a small, usable deployment
for validating the entire studio. Neural tracking on a GPU uses the native setup
below, or a separately built CUDA image with the matching PyTorch runtime; do not
claim the baseline image supplies neural inference.

The web image uses Next.js standalone output. `NEXT_PUBLIC_API_BASE_URL` is empty
**at build time**, so the browser calls `/api/...` on its own origin. Set another
URL only when building a deliberately separate API deployment. Next.js embeds
public environment variables during compilation; changing a container variable
after building does not change that URL.
[Next.js environment documentation](https://nextjs.org/docs/app/guides/environment-variables).

## Remote access and TLS

The gateway binds to loopback by default. For a private remote workstation, keep
that binding and forward it through SSH:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@your-server
```

For an internet deployment, terminate HTTPS at a trusted host reverse proxy that
forwards to `127.0.0.1:8080`, or configure a domain and automatic TLS in Caddy.
Forward all paths and allow SSE streaming without buffering. Keep authentication
on every path and do not publish API port 8010 or web port 3000 directly. Basic
authentication requires HTTPS when crossing an untrusted network.
[Caddy authentication documentation](https://caddyserver.com/docs/caddyfile/directives/basic_auth).

The gateway limits request bodies to 513 MB, including multipart overhead; the API
independently caps the video payload with `ROTOSTREAM_MAX_UPLOAD_MB`. If you raise
the API cap, raise `request_body.max_size` in `deploy/Caddyfile` accordingly.
Authentication runs before the upstream handles an upload. The API port used for
native development has no authentication: keep it on loopback or put the same
authenticated gateway in front of it.

## Native GPU deployment

Create a Python 3.12+ virtual environment and install `api/requirements.txt` and
`ml/requirements.txt`. Install a PyTorch/torchvision build that matches your CUDA
driver from the [official PyTorch installer](https://pytorch.org/get-started/locally/).
Then install `api/requirements-model.txt` for the pinned model adapter dependencies.
Copy `.env.example` to `.env`, select `ROTOSTREAM_DEFAULT_MODEL=sam2_memory`, and
run `pnpm install --frozen-lockfile` followed by `pnpm dev` for local development.
The first neural request downloads `facebook/sam2.1-hiera-tiny` unless it is cached.
Preload weights on the deployment host before admitting users.

For a production native service, run uvicorn **without reload and with one worker**:

```bash
cd api
../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --workers 1 --no-server-header
```

On Windows use `../.venv/Scripts/python.exe`. Build the web app using an empty
`NEXT_PUBLIC_API_BASE_URL` for an authenticated same-origin proxy, then run the
generated standalone server or `pnpm --dir web start`. Use an OS service manager
to restart both processes. `pnpm dev` is a development launcher, not a service
manager.

## Health, limits and recovery

- `GET /api/health` reports liveness and runtime capabilities.
- `GET /api/ready` checks ffmpeg/ffprobe, workspace writeability and default tracker
  usability. It returns 503 if a prerequisite is missing, without fetching weights.
- The queue is bounded by `ROTOSTREAM_MAX_PENDING_JOBS`; requests beyond capacity
  receive 503 and can be retried. `ROTOSTREAM_JOB_HISTORY` bounds terminal history.
- Preview and background inference share one compute gate. Preview cache size and
  ffmpeg deadline are configurable through `ROTOSTREAM_PREVIEW_CACHE_SIZE` and
  `ROTOSTREAM_FFMPEG_TIMEOUT_S`.
- A completed session is immutable. Re-tracking creates a new session so masks
  being read or exported cannot silently change underneath a request.
- Cancellation is cooperative. The running frame or ffmpeg operation may need to
  finish before the job notices cancellation; the ffmpeg deadline bounds subprocess
  execution.
- On restart, unfinished durable records are marked failed. In-memory job IDs
  disappear; reload the studio, inspect the durable records, and retry the operation.

Use `docker compose --env-file .env.production ps` and
`docker compose --env-file .env.production logs --tail 100 api web gateway` to
diagnose startup. Readiness does not establish that downloaded neural weights fit
the GPU or that a full inference pass succeeds; run the smoke/contract checks too.

## Backup, restore and updates

The `rotostream_workspace` volume contains originals, frames, sessions and exports.
Back up that volume while the API is stopped so metadata and masks form a consistent
snapshot. Retain `.env.production` separately in secure storage. Monitor volume
space; processed frames and exports can be much larger than their original videos.
Delete unwanted clips through the studio to remove their associated artifacts.

```bash
docker compose --env-file .env.production stop api
docker run --rm -v rotostream_workspace:/source:ro -v "$PWD/backups:/backup" alpine:3.22 tar -czf /backup/workspace.tar.gz -C /source .
docker compose --env-file .env.production start api
```

Create `backups/` first. On restore, stop the stack, extract the snapshot into the
workspace volume, preserve ownership UID/GID 10001, and start the stack. Validate
readiness and open an existing session before resuming edits. Do not run `down -v`
unless you intend to delete all stored videos and exports.

Before upgrading, record the deployed Git commit, back up storage, run the checks
in the README, and rebuild with `up --build -d --wait`. To roll back, check out the
previous release and rebuild against the backed-up workspace. No database migration
is required by this release. Multiple API workers or replicas sharing this volume
are unsupported because job ownership and compute locks are local to a process.

## Verification

`pnpm verify` runs lint, TypeScript, frontend behavior tests, Python tests and the
production web build. `pnpm smoke` drives a running native API over real HTTP through
upload, extraction, prompting, tracking, SSE, all six exports and cleanup. The CI
container job additionally builds the images, starts the authenticated gateway,
checks unauthorized access, and runs the same workflow through that gateway.

Container manifests were statically checked locally. Docker Desktop was started
on the Windows validation host, but image-list and pull commands hung, so no local
container execution pass is claimed. Treat the CI container result as the
deployment gate, not static validation or this document.
