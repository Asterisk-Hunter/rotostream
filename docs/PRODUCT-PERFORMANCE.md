# Product workflow performance

RotoStream should make the editor's work feel direct: upload a clip, click the
subject, see whether the mask is usable, and get a deliverable without managing
temporary files or guessing what a long-running job is doing. This page records
what that workflow costs today and how to measure it on a deployment.

## A local end-to-end reference

On 2026-10-06, the HTTP smoke runner processed the same 10.01-second,
640×360, 29.97 fps clip (300 frames) through an isolated local API on CPU using
the `naive` tracker. It uploaded and extracted the clip, generated a prompt
preview, tracked the clip over SSE, fetched review assets, rendered all six
deliverables sequentially, downloaded them, and deleted the test upload.

| Stage | Time |
| --- | ---: |
| Upload response | 0.050 s |
| Frame extraction | 0.461 s |
| One-frame preview | 0.366 s |
| Full-clip tracking, including SSE | 4.674 s |
| Review assets | 0.032 s |
| Six exports, sequentially | 22.205 s |
| Download all six outputs | 0.519 s |
| Complete smoke workflow | 31.349 s |

Export times were 11.315 s for transparent WebM, 5.958 s for the lossless RGBA
PNG sequence, 2.610 s for overlay MP4, 1.844 s for replaced-background MP4,
0.315 s for RLE JSON and 0.163 s for mask PNGs. The test confirmed that all
300 frames were tracked and that each output downloaded with the expected
container signature. The PNG sequence was about 78.6 MB; the other five outputs
were between 0.49 MB and 2.85 MB.

These are single local observations on a Windows CPU and a small 360p clip.
They are not Cloud Run timings, p50/p95 latency, GPU measurements, or a promise
about larger and higher-resolution footage. The `naive` tracker is a deliberately
simple reference baseline; this is not a quality or neural-speed result. Export
durations can vary with ffmpeg, CPU load, storage and codec settings.

## Changes aimed at saving editor time

- H.264 video exports now use `veryfast` at the same CRF 18 quality target. In a
  same-machine, same-clip check, replaced-background export moved from 2.612 s at
  `medium` to 2.460 s at `veryfast` (about 6% faster) and the file was 5.5%
  smaller. This is a modest result from one comparison, not a broad speed claim.
- Transparent WebM uses VP9 `cpu-used 6`. A trial at 8 did not show a reliable
  improvement; the measured output size stayed the same and the elapsed times
  were close enough to treat the difference as run-to-run variation.
- Cutout PNG frames use compression level 3 to reduce CPU time while preserving
  exact RGBA pixels. An automated test compares the decoded pixels to the source
  frame and mask. This keeps the lossless promise while reducing time spent
  waiting for a large sequence.
- Video exports stream composed frames to ffmpeg rather than writing and reading
  an intermediate PNG sequence. Timeline classification and review bins are
  memoized, and playback samples at up to eight frames per second while the
  playhead remains in clip time.
- The workflow exposes stage progress, review coverage and export contents so
  the editor can decide what to fix or download instead of waiting without
  feedback or guessing what a file contains.

The encoder preset change is reversible, and output tests cover the media
containers and lossless cutout pixels. Quality settings remain explicit in the
export implementation; benchmark claims should be refreshed if they change.

## Measure a deployment safely

Run this from the repository root in PowerShell. It prompts locally for the
editor password; do not put the password in a command, shell history, or chat.
The smoke script signs in to the Vercel app over HTTPS, keeps its session cookie
in memory, and deletes the uploaded test clip after the run. Choose a foreground
point that lies on the subject in the extracted working resolution. For a fast
check, the included jellyfish clip is 640×360 and the example point below is near
its center.

```powershell
$env:ROTOSTREAM_SMOKE_USER = "editor"
$securePassword = Read-Host "RotoStream editor password" -AsSecureString
$env:ROTOSTREAM_SMOKE_PASSWORD = [System.Net.NetworkCredential]::new("", $securePassword).Password
node scripts/python.mjs api/scripts/smoke.py `
  --base https://rotostream.vercel.app --auth-mode session `
  --file test-media/jellyfish-360p-10s.mp4 `
  --point-x 320 --point-y 180 --frame-index 151 `
  --model naive --export-kind replace_bg
Remove-Item Env:ROTOSTREAM_SMOKE_PASSWORD
Remove-Item Env:ROTOSTREAM_SMOKE_USER
```

The final `PROFILE` JSON line reports upload, extraction, preview, tracking,
review, export and download times. `--export-kind` keeps a deployment run to one
deliverable; omit it to run all six exports. This route exercises the branded
sign-in, Vercel's server-side credential relay, and the Cloud Run API as an editor
uses them. To isolate the Cloud Run gateway instead, set `--base` to the Cloud Run
service URL and omit `--auth-mode session`; the default uses HTTP Basic Auth.
Check the Cloud Run revision, machine type, region and concurrency alongside each
result. Repeat runs before making a performance claim, and test representative
720p/1080p clips before estimating editor wait times.

For a local profile, start the API and use the same command with
`--base http://127.0.0.1:8010`; omit the auth environment variables for an
unauthenticated local API.
