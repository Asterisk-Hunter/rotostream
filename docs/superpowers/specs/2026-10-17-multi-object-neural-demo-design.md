# Multi-object neural demo

Proposed design for review, 2026-10-17. Not implemented by this document.

## Goal and boundary

Strengthen the existing résumé demo: select up to three subjects, correct each
independently, inspect all masks, and export either one subject or their combined
cutout. Enable the existing SAM 2.1 reproduction on supported GPU hardware.
Retain the CPU baseline as an explicitly labelled fallback. Do not add accounts,
billing, multi-user infrastructure or durable queues.

Success means tested multi-object editing and a verified neural deployment path;
it does not mean Meta feature parity, real-time performance, or a new joint DAVIS
result. Existing research protocols and benchmark reports remain unchanged.

## Approach

Recommended: orchestrate independent object tracks above the existing tracker
contract, reuse loaded weights, and expose explicit object selection in the UI.
This preserves causal tracking and bounds VRAM. Alternatives are replacing the
tracker with the upstream joint predictor (weakens the reproduction's purpose),
or redesigning the decoder/contract for joint inference (a separate research task).

Each object has a stable validated ID, label, colour, prompts and independent
scores/quality/memory diagnostics. IDs are never arbitrary filesystem paths.
Track at most three objects sequentially under the existing compute gate. Load
weights once, then reset mutable state and set the video between objects. Each
object builds its own causal plan from its own prompted frames.

## API, storage and compatibility

- Extend tracking requests with `objects`; keep legacy `prompts` as an
  alternative, normalized to one default object. Reject mixed forms, duplicate
  IDs, excess objects, duplicate frame records within an object and objects
  without a valid foreground seed.
- Version new session metadata. Store raw masks under validated per-object
  subdirectories and persist the binary union in the existing session `masks`
  directory so ZIP, RLE and video export paths agree.
- Old requests and saved sessions continue to behave as one-object runs. No
  destructive storage migration. New per-object fields are additive.
- Per-object scores remain the source of identity review. Aggregate problem
  frames are the union of object problems; a scene is sound only if every object
  is sound. Union mask coverage must not hide a missing object. Update the
  response validator deliberately rather than deriving scene quality from a
  single flat score list.
- Export scope is `combined` (default) or `object`, with a validated object ID
  required for object scope. The manifest records scope, identities and policy.
- A failed or cancelled run never publishes a successful partial session. It
  remains non-exportable. Successful sessions remain immutable.

## Overlaps

Combined exports use the OR of raw masks; selected-object exports use that
object's raw mask. Coloured review overlays use stable object-list order to
display overlapping identities. This presentation choice does not alter stored
raw masks or claim to resolve physical occlusion.

The current tracker returns boolean masks and scalar quality, not per-pixel
logits. Do not compare scalar scores as if they were calibrated pixel confidence.
Joint logit arbitration is outside this demo change.

## Editor experience

Add a compact object list with Add, select, rename and remove. Each object has a
consistent colour. Selection controls prompting, undo, score timeline and memory
inspection; review can show all objects together. Label clearing actions so they
cannot silently erase another object's prompts.

Drafts and undo histories are per object and per clip. Preview request identity
includes object ID, clip, frame, model and prompt signature; late responses from
another object are discarded. Reload restores object identities and the prompts
that produced a saved run.

Before export, choose all subjects or one named subject. Retain the completed
MP4 preview and the existing format/audio/dimension descriptions.

## Neural runtime and hosted GPU

First validate the new workflow locally with the available RTX 4050 and cached
SAM 2.1 weights. Measure loading, first preview, tracking, corrections, memory
usage and exports; do not infer cloud latency from local observations.

Build a separate pinned CUDA/PyTorch neural API image with released weights
prefetched. GPU mode must fail readiness clearly when CUDA/weights are unusable;
never silently label a CPU fallback as GPU inference.

Prepare a separate Cloud Run staging service in a supported region. Proposed
L4 configuration: one GPU, at least 4 vCPU/16 GiB, maximum one instance, minimum
zero, bounded clips/objects/queue, and instance-based billing. Verify project
quota and regional availability before activation. Keep active processing local
to the GPU instance where feasible; benchmark storage overhead explicitly.

Preserve the current CPU deployment and workspace for rollback. Validate the
neural staging service through upload, prompts, tracking, review and downloads
before switching Vercel's backend. Never run two writers against the same shared
workspace. GPU use creates a different running-instance cost; no GPU has been
provisioned by this design.

Official constraint reference: [Cloud Run service GPUs](https://docs.cloud.google.com/run/docs/configuring/services/gpu).
GPU services use instance-based billing and can scale to zero; that does not
make active GPU inference free. Exact pricing and quota must be checked for the
selected region at deployment time.

## Acceptance tests

1. Existing single-object API, stored sessions and six exports remain compatible.
2. Two objects prompted on the same frame retain separate masks and identities.
3. Different starting frames and backward propagation preserve each causal plan.
4. Correcting A retains B's draft, identity and prompts; switching clips, undo and
   reload restore the expected object state. Delayed previews never cross objects.
5. Known overlapping masks produce exact union and selected-object pixels across
   PNG, RLE and video paths; manifests identify the actual scope.
6. One lost object remains flagged when another covers every frame.
7. Invalid IDs, ambiguous payloads, missing seeds and excess objects are rejected.
8. Cancellation/failure cannot leave an exportable partial scene.
9. Browser checks cover object creation/selection/rename/removal, corrections,
   review, both export scopes and actual downloaded output at desktop/narrow widths.
10. Local neural checks and staged hosted checks record model, hardware, revision,
    clip, object count, timings and output inspection. Run required existing CI.

## Delivery order

Implement schema/storage/orchestration and overlap tests first, then editor and
export scope, then local neural verification and GPU deployment packaging.
Switch the hosted backend only after the staged neural workflow passes. Retain
clear CPU versus neural labels throughout.
