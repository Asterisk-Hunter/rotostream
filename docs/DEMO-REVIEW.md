# Demo review — 2026-10-15

The target is a résumé demonstration for reviewers, not a public multi-user
service. Private accounts, tenant isolation and durable cloud queues are outside
this demo release. The current workspace is shared; use non-private test footage.

## Comparison with Meta's SAM 2 demo

The [Meta demo](https://sam2.metademolab.com/) was inspected through onboarding.
It starts with a sample video, an instruction to click an object and a selection
tutorial. Its reference implementation documents click/box refinement,
propagation and multiple objects. This review did not run the same clip through
both deployed services, so it establishes no relative quality or speed result.

| User task | RotoStream today | Gap / intended demo experience |
| --- | --- | --- |
| Start editing | Upload or select a shared clip | Cold-start recovery must explain the wait and recover without manual intervention. |
| Mark a subject | Foreground/background clicks, boxes, undo and saved-session prompts | One clearly named subject workflow; explicit tracker choice and limits. |
| Track real footage | Neural reproduction is implemented and evaluated locally; hosted service uses the colour baseline | Hosted baseline is not equivalent to Meta's neural demo. GPU neural hosting is a distinct deployment choice. |
| Track several objects | One object per session, independent-object evaluation | Joint multi-object editing is absent; do not imply parity. |
| Inspect motion | Timeline, sampled review playback, confidence/missing-mask navigation | Review flags do not prove segmentation correctness. Inspect the rendered deliverable too. |
| Obtain output | Six export formats, manifest and download | Completed MP4 exports now have an inline rendered-file preview. |
| Return after restart | Stored sessions/exports survive; active jobs require retry | Acceptable disclosed demo limit; automatic job recovery requires a separate architecture. |

Reference: [Meta SAM 2 video inference documentation](https://github.com/facebookresearch/sam2#video-prediction).
RotoStream's measured DAVIS protocol, initialization exclusions and hardware are
in [RESULTS](RESULTS.md); those numbers cannot be compared directly with an
unmeasured browser demo.

## Hosted run and output inspection

The anonymous Vercel → Cloud Run workflow was exercised using the included
`samples/moving-square.mp4`: 60 frames, 640×360, 15 fps, four seconds. The live
API reported CPU, `naive`, no PyTorch; Cloud Run revision was
`rotostream-00006-9pm`. A 30-second cold startup request timed out and required
Retry. This is an observed failure, not a measured cold-start percentile.

Upload/extraction, foreground prompting at (38, 180), tracking, sampled review,
blur-background MP4, black-background MP4 and download all completed. The server
job records, rounded to seconds, show extraction 10 s, tracking 10 s, blur export
4 s and black export 2 s. Tracking's stored elapsed duration was 9.607 s. These
are job durations from one synthetic workload, not full wall-clock workflow
timings or broad performance claims.

The downloaded blur MP4 decoded all 60 frames at 640×360 / 15 fps / 4.000 s with
H.264 video. Five masks were compared with the known geometric square:

| Zero-based frame | Mask IoU |
| --- | ---: |
| 0 | 0.9893 |
| 15 | 0.9966 |
| 30 | 0.9951 |
| 45 | 0.9971 |
| 59 | 0.9971 |

Side-by-side source/export images were inspected at those frames. Across all
decoded frames, mean absolute RGB differences were 2.338/255 inside the true
square and 3.598/255 outside. The small outside change is expected for this
low-detail synthetic background; it is not a general blur-quality result. Lossy
H.264 and imperfect mask edges allow small subject differences. This easy test
does not establish real-footage, occlusion or fine-edge quality.

Raw observations are in [the JSON record](../reports/product/demo-review-20261015.json).
Local artifacts are in `runs/product-audit-20261015/`: `review.webm` (a verified
52.6-second 1440×900 VP8 recording), the two MP4s, session/job JSON, screenshots,
and sampled source/export comparisons. The recording captures review playback
and a black-background export; upload and the first tracking run occurred before
recording. Large runtime media is intentionally excluded from Git.

## Findings and fixes in this pass

- Startup recovery: retry only transient bootstrap reads with a finite budget;
  retain selection/prompts/model preference and show a useful manual Retry after
  exhaustion. No automatic mutation retries.
- Public visitors should not see deletion controls their access cannot use.
  Server-derived editor capability now controls those buttons; API enforcement
  remains authoritative.
- At 390px, the original header crowded the brand and filename. The narrow
  header now uses a short shared-demo label and keeps the filename out of that
  row. The production build was checked at 390×844 and 1440×900.
- Replace "nothing to fix" with "no automatic review flags"; coverage/confidence
  cannot certify visual correctness.
- Use readable built-in tracker names and permit previewing the actual completed
  MP4 without downloading it first. Video bytes load only when requested.
- Correct public/shared workspace and deletion wording in the field guide and
  distinguish historical validation from current hosted observations.

The baseline remains the largest demonstration limitation on arbitrary footage.
For a neural demonstration, run the documented SAM 2.1 configuration on supported
hardware and show its measured setup/processing times. Do not relabel CPU
baseline results as neural inference.

## Verification of this patch

`pnpm verify` passed: lint, TypeScript, 41 web tests, API/ML tests (with the
existing expected failure) and the production build. The seven startup tests
cover recovery, finite exhaustion, nonretryable errors, timer cleanup, late
responses and user-facing errors. The production dependency audit passed.

A separate production server relayed read-only requests to the hosted demo.
Its health request was deliberately aborted in the browser, then restored.
Startup recovered without pressing Retry and prompt editing became enabled.
At 390px, document scroll width was 380px and anonymous delete controls were
absent. At 1440px, an expanded rendered-video preview played the actual hosted
MP4 and reported 640px video width and a four-second duration. No browser
JavaScript errors were reported in that verification session.

To repeat the browser check, start a production build with a configured API,
abort `**/api/health` with the browser network tool, open the editor, restore
that route before the retry budget expires, and confirm editing recovers without
manual Retry. Then select Export, expand a completed MP4's preview, play it, and
check both desktop and narrow layouts. Do not route mutations into a shared
workspace merely to test startup recovery.
