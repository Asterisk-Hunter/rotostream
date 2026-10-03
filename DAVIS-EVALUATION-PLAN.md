# RotoStream — DAVIS evaluation: review findings and plan

> Historical review from 2026-09-29. The protocol corrections and complete
> 30-clip/61-object evaluation were implemented on 2026-10-03. See
> [current measured results](docs/RESULTS.md); statements below describe the earlier state.

Written 2026-09-29 after an external review of the project. **No code was changed.**
This is a plan for the next piece of work, plus three traps that would otherwise
produce a wrong or misleading number.

Every file:line below was read and verified directly, not inferred.

---

## 1. The correction that unblocks this

The project's own `README.md` marks the DAVIS 2017 val J&F cell as `_pending_`, and
the natural reading is that a number requires training on DAVIS/SA-V first. That
reading is wrong, and it is the most useful thing this review established.

`api/scripts/check_parity.py` loads Meta's released **SAM 2.1** weights into this
repository's memory stack and reports **0 missing, 0 unexpected** weight keys, with
`0.000e+00` max absolute difference on the memory encoder (features and position codes)
and the memory attention output. That is a parameter-for-parameter match, so the
released checkpoint is directly loadable here.

**So the DAVIS number is an inference job, not a training project.** The claim becomes:

> A from-scratch reimplementation of SAM 2's memory mechanism, scoring within *X* J&F
> of the reference on DAVIS 2017 val, with the same weights.

That is a stronger sentence than "I trained on DAVIS and got a number," and it costs
one eval run. It also reframes the target: the question is no longer "can this reach
90.2?" but "how close does an independent implementation land, and where does it
diverge?"

`ml/rotostream_ml/evaluate.py` already accepts `--checkpoint`, so the plumbing exists.

---

## 2. Three traps

### 2.1 The prompt protocol does not match the number you want to compare against

This is the one that would produce a genuinely misleading results table.

The evaluation prompts with **a single interior point**, not the ground-truth mask:

```
ml/rotostream_ml/evaluate.py:263   prompt = sequence.prompt_for(object_index, prompt_frame)
ml/rotostream_ml/sequences.py:139  def prompt_for(self, object_index=0, frame_index=None) -> PromptSet
ml/rotostream_ml/sequences.py:151  return PromptSet(
ml/rotostream_ml/sequences.py:153      points=(PointPrompt(x=point[0], y=point[1], positive=True),)
ml/rotostream_ml/evaluate.py:23      "Prompting follows the semi-supervised protocol: one click on the first frame"
```

`sequences.py:28` defines `interior_point(mask)` — a point derived from the ground
truth, which is correct and reproducible, but it makes this a **1-click** evaluation.

The paper's 90.2 J&F (Hiera-B+) / 90.7 (Hiera-L) is the **𝒢** column of Table 6, which
is explicitly *"based on first-frame ground-truth mask prompts."* The 1-click figures
live in a different table (Table 4), and those are **averaged across 17 datasets**,
not DAVIS-val alone. Its 1-click average for SAM 2 is 64.7.

**Consequence:** a 1-click DAVIS-val number placed next to "paper: 90.2" is comparing
two different protocols. A reviewer will spot it immediately and it will discredit the
whole table.

**Fix:** add a prompt mode to `prompt_for` / `evaluate.py` so the first frame's
ground-truth mask can be supplied as a mask prompt, and record which protocol produced
each number. Until that exists, any DAVIS figure must state its prompt type in the
table itself.

### 2.2 Directional-only memory is a train/test mismatch against Meta's weights

Meta's weights were **trained** with bidirectional conditioning available — a
conditioning frame on either side of the current frame can be attended to. This
repository deliberately forbids that (§2.3 below), so evaluating those weights inside a
directional-only bank is a mismatch between what the model was fitted to use and what
it is permitted to use.

**How much this matters:** under the standard protocol — first-frame prompt, forward
propagation — every conditioning frame is already `< frame_index`, so forward-only
memory is equivalent and the cost should be near zero. That is the case you will
actually run.

**Why it still matters:** if the number comes in well below the reference, "we disabled
half the attention the weights were trained to use" is a candidate explanation *ahead
of* "our implementation has a bug." Know this before spending a day bisecting a defect
that is actually a documented, deliberate design decision.

**Report it as a result, not a caveat.** A directional-only stack scoring *X* against
a bidirectional reference scoring *Y* is exactly the kind of measured trade-off that
turns the deviation in `docs/MODEL_CONTRACT.md` §4 from an assertion into evidence.

### 2.3 The `--bidirectional` flag is not the experiment, and the difference is subtle

The CLI already has a flag that sounds like the ablation one would want:

```
ml/rotostream_ml/evaluate.py:431   "--bidirectional"  "also propagate backwards from the prompt"
ml/rotostream_ml/evaluate.py:284   forward loop:  for index in range(prompt_frame + 1, n_frames)
ml/rotostream_ml/evaluate.py:291   backward loop: for index in range(prompt_frame - 1, -1, -1)
```

That is **two causal sweeps from a single prompt** — a forward pass and a backward
pass, each still clean with respect to its own direction. It is the user-facing toggle
in the studio, and it is *not* a violation of the causality contract. `MODEL_CONTRACT.md`
§4 states this distinction explicitly.

**True bidirectional conditioning** is a different thing: a single forward pass in
which frames *after* the current one inform the prediction, which is what the paper
describes. That is what the contract forbids, and there is currently **no code path for
it** — which is correct, since shipping it would break `test_contract.py`.

**Implication.** To measure the deviation rather than assert it, you need a
deliberately-isolated experiment that can bypass the directional gate without
weakening the shipped model. The good news is the other prerequisite already exists:

```
ml/rotostream_ml/evaluate.py:435   "--prompt-frame"  type=int, default=None
```

`--prompt-frame` lets you prompt mid-clip, which is the *only* protocol under which
bidirectional access can change anything. On the standard first-frame-prompt protocol
it is a no-op, because there is no later conditioning frame to attend to. **Any
bidirectional-vs-directional measurement must use `--prompt-frame`.**

**Verify the paper's claim before leaning on it.** The paper states that prompted
frames may come "from the future" relative to the current frame. That is a statement
about the paper's design; whether the *released implementation* actually permits it in
practice, and how conditioning frames are ordered in the bank, should be confirmed
against `sam2/modeling/sam2_base.py` in `facebookresearch/sam2` before it becomes a
claim in your own write-up. Do not cite it on the strength of a paraphrase.

---

## 3. Extend parity — and expect it to be the interesting one

`check_parity.py` currently covers the memory encoder and the memory attention. It
does **not** cover the mask decoder or the occlusion head, and those are the most
likely places for a real divergence:

- the mask decoder consumes **stride-4 and stride-8** Hiera features through skip
  connections that bypass memory attention entirely
- the occlusion head is an **extra output token** alongside the mask and IoU tokens,
  plus a learned occlusion embedding written back into the bank

Those are the two components with the most implementation freedom, so unlike the
memory path they may **not** reach `0.000e+00`.

**That is not a failure.** A localised, characterised discrepancy in the decoder —
"identical up to the memory stack, differs by *N* in the decoder at *operation X*" —
is a genuinely more interesting finding than a third row of exact zeros, and it is
publishable-grade detail about a reproduction. Budget for it and for the possibility
that you will not match.

---

## 4. The ablation table: which rows actually matter

Seven rows are scaffolded in `README.md` and empty. They are not equally worth running.

| Row | Value | Cost |
|---|---|---|
| memories 1 / 2 / 4 / 6 / 8 | Independently reproduces a published sweep (Table 9c: 73.5 / 73.0 / 73.2). Flat shape = evidence the harness is sound. | Low — flag exists (`Tracker(memory_bank_size=N)`) |
| presence head off | Tests the component with the least validation (the only evidence is one synthetic sequence). | Low |
| object pointers off | Paper says no 9-dataset gain but real SA-V / LVOSv2 gain. | Low |
| temporal pos-emb off | Paper's stated rationale is generalisation to inference-time prompt ranges. | Low |
| **directional vs bidirectional** | **The one that is yours.** Currently an assertion. | Medium — needs an isolated code path + `--prompt-frame` protocol |

Run the cheap four first; they fill the table and independently validate the harness.
Then decide whether the directional measurement is worth a new code path.

---

## 5. Agreed order of work

1. Add a GT-mask prompt mode so the DAVIS comparison is protocol-correct (§2.1).
2. Download DAVIS 2017, run eval with the released checkpoint, get the number.
3. Extend parity to the decoder and occlusion head; characterise any divergence.
4. Fill the cheap ablation rows.
5. If it holds up: the directional-vs-bidirectional measurement via `--prompt-frame`.
6. Only then: make the repo public and add a demo GIF. *(Done 2026-09-29: the repo
   was renamed `NN` → `rotostream` and given a description and topics. Still private,
   deliberately — publish once the results table has a number in it.)*

The order matters at step 6. Publishing while the results table still says
`_pending_` is a weaker position than publishing with a real number and a caveat.

---

## 6. Framing rules that should survive all of this

- **The synthetic 0.96 J&F never goes on a resume or in a summary.** It is a
  regression signal over five exact-GT toy scenes written by the same author as the
  tracker. It stays in the repo with its caveat and it comes out of every headline.
- **Parity proves correctness, not capability.** It answers "is this the same
  architecture?" It does not answer "how well does it segment?" Do not let the two
  substitute for each other.
- **`occlusion` at 0.80 is the honest weak spot** and the most interesting one —
  occlusion and re-entry are exactly what separates a memory tracker from a colour
  matcher. A short analysis of whether the cause is the presence head, the bank, or
  the prompt beats a better average everywhere else.
- **The reused-frozen-Hiera framing stays in the first paragraph.** It is the correct
  and honest position, and stating it proactively is much stronger than being asked.
- **Any number reported is filled from `rotostream_ml.evaluate`'s own output**, never
  typed by hand.

---

## 7. Still outstanding, outside the model

- **`C:\Data\prajwal` has no `.git` and has never been trained.** The LLM
  fine-tuning project is untracked, and untracked work is not a portfolio. The
  instinct to move on to it after DAVIS is right; the instinct to treat it as
  nearly-finished is not.
- **`C:\Data\portfolio-2` has no remote**, and its four case studies are invented.
  That is a live credibility risk independent of anything in this document.
- **`HANDOFF.md` is stale** — it still states that `web/` and `ml/` are not started and
  that the docs are missing. It is the first thing a new reader opens.
