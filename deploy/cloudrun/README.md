# Cloud Run deployment

This setup deploys the web app, API and Caddy gateway as one Cloud Run service.
It is intended for a **single editor** and caps the service at one instance. The
gateway is internet-reachable over HTTPS, while Caddy requires Basic Auth on every
path. Store its bcrypt hash in Secret Manager; do not commit credentials.

The default deployment runs the CPU `naive` tracker. It does not install PyTorch,
download SAM 2.1 weights, or provide GPU inference. The API writes its workspace
through a Cloud Storage volume mount. Cloud Run currently documents these mounts as
a preview feature; Cloud Storage FUSE is not fully POSIX compliant and does not
provide file locking. This is suitable for the single-instance demo, not concurrent
multi-instance writes or a multi-user production service. Active jobs are still
in-memory and are marked failed after an instance restart.

## First deployment

Use PowerShell or another shell with `gcloud` authenticated to the target project.
Choose a region; `asia-south1` is used for the deployed demo. Create a private,
uniform-access bucket, a runtime service account, an Artifact Registry Docker repo,
and a Secret Manager secret for the gateway hash. Grant the runtime service account
`roles/storage.objectUser` on the bucket and `roles/secretmanager.secretAccessor`
on the hash secret. The Cloud Build service account needs Artifact Registry Writer
on the project and access to read the temporary Cloud Build source bucket.

Build the three images from the repository root:

```powershell
gcloud builds submit --config=deploy/cloudrun/cloudbuild.yaml `
  --substitutions="_REGION=asia-south1,_REPOSITORY=rotostream,_TAG=release-1" .
```

Generate the Caddy bcrypt hash with `caddy hash-password` (the repository's Compose
instructions show the container command), then add it as a Secret Manager version.
Copy `service.yaml.template`, replace the project number, region, bucket name,
service-account email, image tags and `__AUTH_SECRET_VERSION__`, then deploy it:

```powershell
gcloud run services replace .\service.yaml --project PROJECT_ID --region REGION
gcloud run services update rotostream --project PROJECT_ID --region REGION --no-invoker-iam-check
```

The service's public invoker check is disabled intentionally because the Caddy
gateway provides the app's password gate. Keep the password hash in Secret Manager.
The manifest sets one maximum instance, 40 request concurrency slots, and a
one-hour request timeout for uploads, SSE progress, and exports. It pins the hash
secret version; update that version in the manifest when rotating the password so
Cloud Run creates a revision that reads it.

## Operations

Use the service URL printed by `gcloud run services describe rotostream`. The
workspace bucket retains source clips, frames, sessions and exports. Back it up
with Cloud Storage lifecycle/versioning or a scheduled backup before relying on it
for important work. Monitor bucket growth and Cloud Run request duration. Cloud
Storage FUSE adds latency to per-frame file access; short clips are best for this
demo. Cloud Run may scale the instance to zero after inactivity, and active jobs
do not survive a restart.

The deployed service was exercised with a 10-second 360p clip: upload,
300-frame extraction, a completed CPU-baseline tracking session, mask retrieval,
and a downloadable RLE mask export. Neural SAM 2.1 requires a separate GPU image
and a region/quota that supports Cloud Run GPUs.
