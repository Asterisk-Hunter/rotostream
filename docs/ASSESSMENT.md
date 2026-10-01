# RotoStream — engineering assessment

**Date:** 2026-09-29 · **Commit:** `574827b` · **Status:** private, not released

A candid read of what this repository proves, what it does not, and where it is
weak. Written to be useful to a reviewer who has not read the code, and to anyone
picking the repo up cold. It is deliberately more negative than `README.md`, because
the README's job is to explain the project and this one's job is to stress-test it.

Provenance is marked throughout. Anything marked **verified** was re-run on the date
above by the author of this file. Anything marked **reported** comes from
`README.md` / `docs/MODEL_CONTRACT.md` and was not independently re-measured here.

---

## 1. What this repository is

An AI rotoscoping studio — click an object once in a video, the tracker propagates a
mask across every frame, export an alpha matte, a transparent video, or a replaced
background — built around a from-scratch reimplementation of the memory mechanism in
**SAM 2: Segment Anything in Images and Videos** (Ravi et al., ICLR 2025,
[arXiv:2408.00714](https://arxiv.org/abs/2408.00714)).

**The honesty line, which the README also states up front.** The hierarchical **Hiera
image encoder is reused, pretrained and frozen**; it is not a contribution of this
project. The **memory encoder, memory bank, memory attention, mask decoder and
occlusion/presence head are built here**. This is a reproduction, not a novel
contribution, and no number in this repository should be read as a claim to improve on
the paper.

## 2. The seam

`api/app/models/base.py` is the only file a model author needs. A plugin implements:

```
load(device, checkpoint) → set_video(FrameSource) → add_prompt(PromptSet)
                         → repeated propagate(frame_index, Direction)
```

plus optional `reset()`, `memory_state()` (which powers the UI memory inspector) and
`warmup()`. `TrainableTracker` adds `trainable_parameters()` and
`training_step(TrainingSequence)` so the provided harness can drive the model without
touching its internals.

Seven rules are enforced mechanically in `api/tests/test_contract.py`. The load-bearing
ones: masks are `(H, W)` bool at `FrameSource` resolution, not at the encoder's internal
resolution; `object_present=False` requires an all-`False` mask; and `propagate` may
condition only on frames already visited in its direction.

## 3. The model — implemented, and bit-exact

`api/app/models/sam2_memory.py` (579 lines) plus `api/app/models/sam2_stack/` (encoder
wrapper, memory attention, two-way decoder, bank, losses).

**Verified 2026-09-29** by running both gates:

| Gate | Result |
| --- | --- |
| `api/scripts/check_parity.py` — weight load into this stack | 0 missing, 0 unexpected keys |
| `api/scripts/check_parity.py` — memory encoder features + position codes | max abs diff `0.000e+00` |
| `api/scripts/check_parity.py` — memory attention output | max abs diff `0.000e+00` |
| `api/scripts/check_model.py sam2_memory` | 8 passed, 0 failed, 0 skipped |

Parity means the implementation is **weight-compatible** with Meta's released
checkpoint. Zero missing keys is the strong form of that claim: the two are
parameter-for-parameter the same architecture, so they are interchangeable, not merely
similar. Very few reimplementations can make this claim at all.

**Reported** (from the README, not re-measured here):

- Synthetic evaluation, mean **J&F 0.9600** over 136 frames. `linear`, `distractor`,
  `color_shift`, `reentry` all 1.0000; `occlusion` **0.8000**. Presence head over the
  same run: accuracy 0.9559, precision 1.0000, recall 0.9520.
- `leakcheck --all` — 5 scenarios × 2 directions, all pass.
- `train --dry-run` — gradients through 305 tensors, 274 non-zero.
- `train --overfit --steps 50` — mean loss 0.011446 → 0.000000.

## 4. The one part that is not a reproduction

**The paper permits bidirectional memory.** Its own text states that prompted frames may
come "from the future" relative to the frame being predicted, so a prompted frame
anywhere in the clip can be attended to.

**This repository forbids it.** `FORWARD` may read only indices `< frame_index`;
`BACKWARD` only indices `> frame_index`. The rule is enforced twice: by
`test_contract.py`, and by `ml/rotostream_ml/leakcheck.py`, which *poisons every
unvisited frame with noise and requires the mask at the probe frame to be unchanged*.

The reason is that bidirectional prompting inflates J&F and does not describe the
behaviour a user gets from click-then-propagate. It is a small but real methodological
contribution, and it is why any number this repository produces should be more
trustworthy than the paper's interactive figures. It is documented as a deliberate
deviation in `README.md` and `docs/MODEL_CONTRACT.md` §4, and the model must not
adopt bidirectional memory to "fix" it.

## 5. Surrounding machinery

**Harness** (`ml/rotostream_ml/`, ~93 tests): `metrics.py` implements J (IoU), F
(boundary via Martin's `seg2bmap`) and J&F under the DAVIS protocol, aggregated as a
sequence mean rather than a frame pool. `davis.py` loads DAVIS 2017. `synthetic.py`
builds five exact-ground-truth scenarios. `evaluate.py` resolves trackers through the
plugin registry and goes through the contract, never a bespoke inference path.
`train.py` owns the optimizer, schedule, accumulation, checkpointing and logging, and
reaches only the `TrainableTracker` public surface.

**Application:** FastAPI with routers for health, models, videos, tracking and exports;
an ffmpeg pipeline; an in-process thread job manager with an exclusive gate; and a warm
tracker cache keyed by (video, model, checkpoint) so interactive preview does not reload
weights on every click. Frames are pre-extracted to JPEG at a working resolution (long
side 960). Six export formats: `alpha_webm`, `overlay_mp4`, `replace_bg`, `cutout_zip`,
`mask_zip`, `mask_rle_json`.

**Studio** (Next 16 / React 19 / TypeScript / Tailwind 4): dropzone and project rail,
click-to-prompt on the frame canvas (left = positive, right/alt/shift = negative), the
server-rendered RGBA mask stacked as an image, a timeline with per-frame confidence
bars, a model picker fed by `GET /api/models`, the **memory-bank inspector**, the export
panel, and SSE job progress with a polling fallback.

**Environment:** Windows. The venv is `--system-site-packages` on purpose, so
`torch` 2.8+cu128, OpenCV and SciPy come from the global Python 3.13 rather than
re-downloading ~2.5 GB.

## 6. What is not done

- **DAVIS 2017 val J&F is absent.** The README marks it `_pending_` and explains why
  that is correct rather than a placeholder: a number is only meaningful after a real
  training run, and inventing one would defeat the purpose. This is the single largest
  gap in the repository.
- **The ablation table is empty.** Seven rows are scaffolded — memories 1/2/4/6/8,
  presence head off, temporal positional embedding off, object pointers off. The
  constructor flags exist; no numbers have been produced. The paper's own memory-size
  sweep is essentially flat (4/6/8 → 73.5/73.0/73.2 J&F), so this table is where a
  reader would most expect independent confirmation.
- `SessionOut.scores` returns a score for every frame, which is a large payload on a
  long clip. A `?include_scores=false` option is sketched but deliberately not shipped,
  because the frontend timeline consumes `scores` today.
- The DAVIS `download()` path uses `urllib` with no retry, checksum or resume.
- One tracked object per session, by design: multiple objects would need per-object
  memory banks.

## 7. Where it is weak

Stated plainly, because a reviewer will find these anyway.

1. **Parity proves correctness, not capability.** The bit-exact result answers "is this
   the same architecture?" It does not answer "how well does it segment?" A reader who
   stops at parity will be right to ask what the thing actually scores.
2. **The synthetic 0.96 is not a benchmark result**, and the README says so. Five
   exact-GT toy scenes were written by the same author as the tracker. It is a
   regression signal. Quoted without that caveat it reads as cherry-picking, which is
   why it should never appear in a summary of results.
3. **`occlusion` at 0.80 is the honest weak spot** — and the most interesting one.
   Occlusion and re-entry are precisely what separates a memory tracker from a colour
   matcher, so the one scenario that scores lowest is the one that matters most. The
   presence head is the least-validated component, since the only evidence for it is a
   synthetic sequence and an accuracy figure computed on the same run.
4. **The ablation table is a promise the repository has not kept.** Seven empty rows
   read worse than no table, and a reader may reasonably assume the ablations were run
   and were unflattering.
5. **Directional-only memory is a real trade-off, not purely a win.** The paper's
   bidirectional design is more capable; this repository is more honest. The cost is
   that a prompt on frame 300 cannot inform frame 12, which the paper's design allows.
   The claim is defensible, but it is a narrowing, and it should be argued rather than
   assumed.

## 8. Implementation scars worth preserving

`docs/MODEL_CONTRACT.md` §5.1 records seven corrections that only surfaced while
building, none of which are derivable from the paper. The most expensive:

- The reference implementation is sequence-first `(seq, batch, ch)`; this stack is
  batch-first `(batch, seq, C)`. Transposing the argument makes the reference's RoPE
  key-side repeat compute `k_len // q_len` and materialise a 16 GB tensor. It crashed
  the development machine rather than raising an exception.
- Attention must not be materialised. With a full bank an explicit `q @ k.T` is roughly
  3 GB *per layer* in float32; `F.scaled_dot_product_attention` is both smaller and
  bit-exact against the reference's eager path on CPU float32.
- Inference must run under `torch.no_grad()`. The bank retains features and position
  codes derived from decoder outputs, so an un-detached graph is pinned by the bank and
  grows with the clip. The symptom is an access violation, not a Python exception.
- Gradient checkpointing is required to fit a consumer GPU, because the memory
  encoder's stride-2 pass over a 1024×1024 mask is ~0.25 GB of activation per frame.

These are worth keeping because each is a trap that a correct-looking implementation
falls into silently.

## 9. How to verify all of the above

```bash
# the correctness claim
.venv/Scripts/python api/scripts/check_parity.py

# the contract
.venv/Scripts/python api/scripts/check_model.py sam2_memory

# causality
cd ml && python -m rotostream_ml.leakcheck --model sam2_memory --all

# quality (synthetic — a regression signal, not a benchmark)
cd ml && python -m rotostream_ml.evaluate --model sam2_memory --dataset synthetic

# the whole suite
.venv/Scripts/python -m pytest api/tests ml/tests
```

`check_parity.py` needs to download the released SAM 2.1 weights on first run. The
leakcheck suite re-runs the tracker twice per probe and is the slowest of these; on CPU,
bound it with `--max-checks 3`.

## 10. What to do next, in order

1. **Train on DAVIS 2017 and fill in the number.** Any honest figure beats `_pending_`.
   A modest result reported plainly, with the paper's 90.2 quoted as context rather
   than a target, is the whole artifact.
2. **Run the memory-size ablation** at minimum. It is the table the paper already
   published, so reproducing its shape is independent evidence the harness is sound.
3. **Say more about `occlusion` 0.80.** A short analysis of where the mask fails on
   the synthetic occlusion clip — and whether the cause is the presence head, the
   memory bank, or the prompt — is worth more than a better average.
4. **Add a real video to the repository.** There is no sample media, so the studio
   cannot be run by anyone who does not already have a video file to hand.
5. Only then consider the product surface: onion skinning, keyboard-first frame
   stepping, multi-object sessions. The model is a solved problem here; the craft
   around it is not.
