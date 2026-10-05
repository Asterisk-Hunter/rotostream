# Vercel frontend deployment

Vercel hosts the Next.js studio from `web/`. The Python API and video-processing
jobs stay on Cloud Run, where the long-running tracker, uploads, SSE progress, and
persistent workspace already run. The Vercel Proxy serves a branded login page,
validates the editor session cookie, then forwards `/api/*` requests to Cloud Run
with Basic Auth attached server-side. The browser never sees a browser-native
credential prompt, and the password never enters the frontend bundle.

## Configure

Link the `web/` directory to a Vercel project with `web/` as its Git root
directory, then add these Production environment variables:

```text
ROTOSTREAM_BACKEND_URL=https://<Cloud-Run-service-url>
ROTOSTREAM_AUTH_USER=editor
ROTOSTREAM_AUTH_PASSWORD=<the editor password, stored as a sensitive secret>
```

The Vercel project must keep the root directory set to `web`. The API base URL is
same-origin in production, so do not set `NEXT_PUBLIC_API_BASE_URL`. The session
signing key is derived from the editor password and does not need a separate
environment variable. Add these three variables to Preview only if preview
deployments should be usable; otherwise keep previews private. Keep the production
password out of untrusted preview builds.

Deploy from `web/` with `vercel --prod`, or connect the Git repository and deploy
the production branch. Studio pages and API calls require the eight-hour signed
session cookie. Cloud Run retains its Basic Auth gateway; Vercel attaches those
credentials only to server-side API rewrites. Static Next.js build assets are
public and contain no workspace data.

API traffic uses an external-origin rewrite instead of a Vercel Function. This
keeps video uploads out of Vercel Function request-size and execution-time limits;
Cloud Run continues to enforce its configured upload cap and request timeout.

## Verify

- An unauthenticated page request redirects to `/login` without a Basic Auth challenge.
- A successful login sets an HTTP-only cookie and opens the studio.
- An unauthenticated `/api/health` request returns JSON `401`; an authenticated
  request reaches Cloud Run through the server-side credential relay.
- Upload, SSE tracking, frame/mask retrieval, and export pass through the Vercel
  rewrite and complete on Cloud Run.

Cloud Run remains the processing service; this deployment does not run the Python
API or SAM 2 inside Vercel.
