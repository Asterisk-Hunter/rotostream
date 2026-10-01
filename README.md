# RotoStream

**A reproduction of SAM 2's memory mechanism, wrapped in a rotoscoping studio.**

**What is reused and what is built — read this before the results.** The
hierarchical Hiera image encoder is **reused, pretrained and frozen**; it is not a
contribution of this project. Everything that *is* SAM 2's contribution — the
**memory encoder, the memory bank, the memory attention, the mask decoder and the
occlusion/presence head** — is **implemented from scratch** in this repository
(`api/app/models/`). Nothing below is an achievement of the frozen encoder.

RotoStream is a from-scratch reproduction of the memory-attention video object
tracker in *SAM 2: Segment Anything in Images and Videos* (Ravi et al., ICLR 2025,
[arXiv:2408.00714](https://arxiv.org/abs/2408.00714)): click an object once and
the tracker propagates its mask across every frame.

This is a **reproduction**, not a novel contribution. The paper is the source of
the architecture; the surrounding application, the harness and the causality
tightening are this repo's engineering work.

---

## Results

Two different questions, two different kinds of evidence. Neither is a substitute
for the other: parity says the implementation is *correct*, the benchmarks say
what it *does*.

### 1. Is this actually SAM 2's memory mechanism?

Yes, to the last bit, and one command shows it:

```bash
.venv/Scripts/python api/scripts/check_parity.py
```

It loads the released SAM 2.1 weights into this repository's memory stack and runs
both implementations on identical inputs:

| Comparison | Result |
| --- | --- |
| Weight load | 0 missing, 0 unexpected keys |
| Memory encoder (features, position codes) | max abs diff `0.000e+00` |
| Memory attention (conditioned features) | max abs diff `0.000e+00` |

That is the claim the rest of this README rests on. It is checked, not asserted.

### 2. What quality comes out, and is it leak-free?

Zero-shot — the memory stack above is warm-started from the released weights and
never fine-tuned — on the synthetic scenarios in `ml/rotostream_ml/synthetic.py`:

```bash
cd ml && python -m rotostream_ml.evaluate --model sam2_memory --dataset synthetic
```

| Sequence | Frames | J | F | J&F |
| --- | --- | --- | --- | --- |
| color_shift | 24 | 1.0000 | 1.0000 | 1.0000 |
| distractor | 24 | 1.0000 | 1.0000 | 1.0000 |
| linear | 24 | 1.0000 | 1.0000 | 1.0000 |
| reentry | 34 | 1.0000 | 1.0000 | 1.0000 |
| occlusion | 30 | 0.8000 | 0.8000 | 0.8000 |
| **mean** | **136** | **0.9600** | **0.9600** | **0.9600** |

Presence head over the same run: accuracy 0.9559, precision 1.0000, recall 0.9520.
The occlusion sequence is the one that separates a memory tracker from a colour
matcher, and it is the one where the score drops.

These are synthetic scenes chosen to exercise memory (linear motion, occlusion,
re-entry, a colour-matched distractor), **not** DAVIS. They are a regression
signal, not a leaderboard number.

Causality is checked separately, on every scenario in both directions:

```bash
cd ml && python -m rotostream_ml.leakcheck --model sam2_memory --all
```

Masks are recomputed with the unvisited frames altered; if any mask changes, the
run fails. All 5 scenarios pass in both directions.

### 3. DAVIS 2017 val J&F — pending

Deliberately still blank. A DAVIS number is only meaningful after a real training
run on DAVIS/SA-V, and this checkout has neither the dataset nor a finished run;
inventing one would defeat the point of the project. Fill this in from
`rotostream_ml.evaluate`'s own output, never by hand.

| Backbone | DAVIS 2017 val J&F | FPS | Notes |
| --- | --- | --- | --- |
| Hiera-T | _pending_ | _pending_ | `--json runs/davis_val.json` |
| Hiera-B+ | _pending_ | _pending_ | paper reports 90.2 / 43.8 |
| Hiera-L | _pending_ | _pending_ | paper reports 90.7 / 30.2 |

Paper reference points (SA-V-trained SAM 2.1, for context only — not targets):
90.2 J&F on DAVIS 2017 val with Hiera-B+ at 43.8 FPS, 90.7 with Hiera-L at 30.2 FPS.

### 4. Training

`ml/rotostream_ml/train.py` drives the stack through `TrainableTracker`:

```bash
cd ml
python -m rotostream_ml.train --model sam2_memory --dry-run   # wiring, 3 steps
python -m rotostream_ml.train --model sam2_memory --overfit --steps 50
```

The dry run checks the failure modes that waste a day — no trainable parameters, a
detached loss, a branch with no gradient — and reports gradients flowing through
305 tensors (274 of them non-zero). The overfit check fits a single clip and fails
if the loss did not fall; it takes the mean loss from **0.011446 to 0.000000**, a
100% drop over 50 steps at lr 1e-4.

Gradient checkpointing is on by default during training (`MemoryStack.set_gradient_checkpointing`):
a clip is one graph, and the memory encoder's stride-2 pass over a 1024×1024 mask
is ~0.25 GB of activation *per frame*, which is what makes clip length rather than
model size decide whether a step fits. Without it a 16-frame step sat at the 6 GB
ceiling of a laptop RTX 4050 for over ten minutes; with it, steps are ~10-15 s.

### Ablations (skeleton)

Produced by `python -m rotostream_ml.evaluate` with the corresponding
`memory_bank_size` / flags. Empty until run — the flags exist
(`Tracker(memory_bank_size=N)`), the numbers do not.

| Variant | J&F | Δ |
| --- | --- | --- |
| memories = 1 | | |
| memories = 2 | | |
| memories = 4 | | |
| memories = 6 (default) | | |
| memories = 8 | | |
| presence head off | | |
| temporal pos-emb off | | |
| object pointers off | | |

The paper's memory-size sweep (Table 9c) is essentially flat — 4 / 6 / 8 give
73.5 / 73.0 / 73.2 J&F — which is why 6 is the default here.

---

## Quickstart

```bash
# 1. Python env (Windows paths; swap for bin/ on POSIX).
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r api/requirements.txt -r ml/requirements.txt

# 2. Run API + web together (prefixed logs, one Ctrl-C stops both).
pnpm dev
#   API  -> http://127.0.0.1:8010/docs
#   web  -> http://localhost:3000
```

`ROTOSTREAM_DEFAULT_MODEL` selects the tracker (default `naive`), and
`ROTOSTREAM_DEVICE=auto|cuda|cpu`. Every setting is listed in `.env.example`. The
web app reads `NEXT_PUBLIC_API_BASE_URL` (`web/.env.example`).

The surrounding app already runs end to end today: upload a video, click to
prompt, propagate, and export in six formats.

## Verify

```bash
.venv/Scripts/python -m pytest api/tests ml/tests   # contract, API, metrics, harness
.venv/Scripts/python api/scripts/check_model.py     # per-tracker contract check
.venv/Scripts/python api/scripts/check_parity.py    # numerical parity vs the released weights
cd web && pnpm exec tsc --noEmit                    # frontend types
```

`check_model.py` prints one line per check and is the fastest loop while working
on a model. `check_parity.py` is the correctness claim from §1 above.
`python -m rotostream_ml.leakcheck --model sam2_memory --all` proves the memory
bank never reads unvisited frames.

---

## How it is put together

| Piece | Where |
| --- | --- |
| Tracker contract (the seam) | `api/app/models/base.py` |
| Memory stack | `api/app/models/sam2_stack/` (modules, decoder, bank, losses) |
| Tracker glue (contract + training step) | `api/app/models/sam2_memory.py` |
| Reference baseline (no memory) | `api/app/models/naive.py` |
| HTTP API, storage, jobs, ffmpeg | `api/app/` |
| Evaluation + training harness | `ml/rotostream_ml/` |
| Studio UI | `web/src/` |

**Memory stack, in one paragraph.** The frozen Hiera encoder produces
multi-scale features once per frame. A memory encoder conv-downsamples the
predicted mask and sums it with those *unconditioned* features — no second image
encoder is run. Two FIFO queues hold recent and prompted frames (projected to 64
channels); 4 memory-attention blocks with 2D spatial RoPE let the current frame
cross-attend over that bank and over 4×64-dimensional object pointers. A SAM-style
two-way mask decoder consumes stride-4 and stride-8 Hiera features directly
through skip connections, and an occlusion head reports `object_present` while an
occlusion embedding is written back into the bank.

### A deliberate deviation from the paper

The paper's memory is **bidirectional**: a prompted frame may be attended to even
when it lies "in the future" of the frame being predicted. This repo is
**stricter and directional** — `FORWARD` may read only indices `< frame_index`,
`BACKWARD` only `> frame_index` — so a benchmark number reflects the
click-then-propagate behaviour a user actually gets. The rule is enforced by
`api/tests/test_contract.py` and `ml/rotostream_ml/leakcheck.py`. See
`docs/MODEL_CONTRACT.md` §4; the model must not adopt bidirectional memory.

## Documentation

- [`docs/MODEL_CONTRACT.md`](docs/MODEL_CONTRACT.md) — the model author's reference: contract rules, shapes, the memory-stack spec, the three correctness corrections, and the verification order.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — request lifecycle (upload → extract → prompt → propagate → overlay → export), disk layout, jobs and the leak-free step planner.
- [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md) — candid read of what this repository proves, what is missing, where it is weak, and what to do next. Start here if you want the honest version.

## License

The reference implementation and checkpoint referenced above are Apache 2.0
(facebookresearch/sam2).
