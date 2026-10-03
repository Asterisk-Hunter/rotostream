# RotoStream studio

The Next.js studio drives the FastAPI video tracker: upload a clip, mark an object,
track it across frames, inspect confidence and memory, then download an export.

## Run locally

Use Node.js 22.6 or newer and pnpm 10.4.1. From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

The studio opens at `http://localhost:3000`; the API defaults to
`http://127.0.0.1:8010`. To run only the frontend, use `pnpm --dir web dev`.
Copy `.env.example` to `.env.local` when the API runs elsewhere.

`NEXT_PUBLIC_API_BASE_URL` is a build-time browser setting. An empty value uses
same-origin `/api` requests, including frame images, SSE progress and downloads;
this is the configuration used by the authenticated reverse proxy deployment.
Rebuild after changing this setting.

## Studio controls

- Click or tap the frame in **Mark object** mode to add foreground points.
- Select **Exclude background**, right-click, or hold Alt/Shift to exclude a region.
- Focus the frame with Tab, move the cursor with arrow keys, and press Enter to
  add a point. Shift+Enter excludes background. Outside the focused frame, left
  and right arrow keys scrub the clip; the frame slider also supports keyboard navigation.
- Prompt multiple frames to correct drift. **Track both directions** includes
  frames before the first prompt. A new run creates a new immutable session.
  With the color baseline, mark foreground on each corrected frame before adding
  background exclusions.
- The timeline shows prompted frames, propagated confidence and reported absence.
- Downloads use the last successful tracking session. Delete requires a second
  click because it removes the source, masks and exports together.

Extraction resumes monitoring after a page reload. Completed sessions and exports
are stored by the API. Unsaved prompt clicks exist only in the current tab.
Progress continues through a polling fallback if SSE disconnects, transient
network errors retry, and a missing job after server restart permits a new run.

## Verify

```sh
pnpm --dir web test
pnpm --dir web lint
pnpm --dir web typecheck
pnpm --dir web build
```

The dependency-free Node test runner checks coordinate scaling and edge pixels,
timecode rollover, job parsing, transient failure recovery, stale responses,
terminal states and cleanup. Browser validation covers the integrated API flow;
these unit tests do not replace it.

## Production

`next build` emits `.next/standalone/web/server.js`, with the repository as the
tracing root. The repository Dockerfile supplies static assets and runs this
standalone server. See `../docs/DEPLOYMENT.md` for the reverse proxy, persistent
API workspace and deployment configuration.

The API is the source of truth for the types in `src/lib/types.ts`. Its model
contracts and architecture are documented in `../docs/`.
