# Vercel frontend deployment

Vercel hosts the Next.js studio from `web/`. The Python API and video-processing
jobs stay on Cloud Run, where the long-running tracker, uploads, SSE progress, and
persistent workspace already run. The studio is open for outreach visitors. The
Vercel Proxy forwards `/api/*` requests to Cloud Run with Basic Auth attached
server-side; the password never enters the frontend bundle.

Share the stable production domain, [`https://rotostream.vercel.app`](https://rotostream.vercel.app),
and its public [`/docs` field guide](https://rotostream.vercel.app/docs). Per-deployment
Vercel URLs may require a Vercel team account and can show Vercel's login screen
instead of the public RotoStream studio.

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
the production branch. Studio pages and API calls are public. Cloud Run retains
its Basic Auth gateway; Vercel attaches those
credentials only to server-side API rewrites. Static Next.js build assets are
public and contain no workspace data.

API traffic uses an external-origin rewrite instead of a Vercel Function. This
keeps video uploads out of Vercel Function request-size and execution-time limits;
Cloud Run continues to enforce its configured upload cap and request timeout.

## Demo limits

The proxy allows 3,000 API requests per minute (including playback frames), two uploads per hour, and 20 other
POST requests per hour per visitor IP. Each proxy instance also allows at most
20 uploads and 100 other POST requests per hour across visitors. Requests beyond
these limits receive JSON `429` with `Retry-After`; rejected requests do not reach
Cloud Run. Mutations require a matching `Origin` header. Public visitors cannot
delete clips or exports; an existing valid editor session can still do so.

These counters live in memory in each Vercel instance. They reset on restart and
are not shared across instances, so they mitigate casual abuse but **are not a
hard billing cap**. Use a durable shared quota or Vercel Firewall rate limiting
for stricter cost control before opening this to broad traffic. The Cloud Run
single-instance cap, upload limits, frame limits and queue capacity remain in
place. This is a shared demo workspace: uploaded clips can be seen by other
visitors; do not use private footage.

Vercel supplies and overwrites `x-forwarded-for` at its edge. On other hosts the
proxy uses one shared bucket rather than trusting arbitrary incoming IP headers.

## Verify

- An unauthenticated page request opens the studio; `/login` redirects to `/`.
- An unauthenticated `/api/health` request reaches Cloud Run through the server-side
  credential relay.
- A POST with an unrelated or absent `Origin` returns `403`.
- A third upload in the same hour returns `429` and `Retry-After`.
- Upload, SSE tracking, frame/mask retrieval, and export pass through the Vercel
  rewrite and complete on Cloud Run.

Cloud Run remains the processing service; this deployment does not run the Python
API or SAM 2 inside Vercel.
