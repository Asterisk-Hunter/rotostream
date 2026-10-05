# Vercel frontend deployment

Vercel hosts the Next.js studio from `web/`. The Python API and video-processing
jobs stay on Cloud Run, where the long-running tracker, uploads, SSE progress, and
persistent workspace already run. The Vercel Proxy protects the studio pages with
the editor password. Next.js rewrites `/api/*` to Cloud Run on the same browser
origin, and Cloud Run's gateway verifies the forwarded credentials. The password
is stored as a Vercel server-side secret; it is never embedded in the frontend
bundle.

## Configure

Link the `web/` directory to a Vercel project with `web/` as its Git root
directory, then add these Production environment variables:

```text
ROTOSTREAM_BACKEND_URL=https://<Cloud-Run-service-url>
ROTOSTREAM_AUTH_USER=editor
ROTOSTREAM_AUTH_PASSWORD=<the editor password, stored as a sensitive secret>
```

The Vercel project must keep the root directory set to `web`. The API base URL is
same-origin in production, so do not set `NEXT_PUBLIC_API_BASE_URL`. Add the same
three variables to Preview only if preview deployments should be usable; otherwise
keep previews private. Keep the production password out of untrusted preview
builds.

Deploy from `web/` with `vercel --prod`, or connect the Git repository and deploy
the production branch. Studio pages are protected by the Basic Auth challenge;
the Cloud Run gateway protects API routes. Static Next.js build assets are public
and contain no workspace data.

API traffic uses an external-origin rewrite instead of a Vercel Function. This
keeps video uploads out of Vercel Function request-size and execution-time limits;
Cloud Run continues to enforce its configured upload cap and request timeout.

## Verify

- An unauthenticated page request returns `401` with a Basic Auth challenge.
- The editor credentials load the studio and allow `/api/health` to reach Cloud Run.
- Upload, SSE tracking, frame/mask retrieval, and export pass through the Vercel
  rewrite and complete on Cloud Run.

Cloud Run remains the processing service; this deployment does not run the Python
API or SAM 2 inside Vercel.
