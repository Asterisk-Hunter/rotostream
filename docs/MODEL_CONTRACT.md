# RotoStream model contract

The single reference for whoever implements the tracker in
`api/app/models/sam2_memory.py`. Everything the application calls, every shape it
hands you, and every number from the paper you need while writing the memory
stack. Read this with `api/app/models/base.py` open — that file is the normative
contract; this one explains it and adds the architecture spec.

Reference: **SAM 2: Segment Anything in Images and Videos**, Ravi et al.,
[arXiv:2408.00714v2](https://arxiv.org/abs/2408.00714) (ICLR 2025).
Code: [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) (Apache 2.0).
The paper's "SAM 2" is this repo's "SAM 2.1" (HF ids
`facebook/sam2.1-hiera-{tiny,small,base-plus,large}`).

---

## 1. The seam

The API, the UI, the export pipeline, the evaluation harness and the training loop
are all written against `api/app/models/base.py`. Implement that interface and the
whole application uses your model with no other edits. Nothing in the rest of the
repo may reach into your internals, and your model must not import the web layer
(`schemas.py`, FastAPI, `storage.py`).

### The seven rules (`base.py`, module docstring)

1. `load` is called exactly once, before `set_video`.
2. `set_video` is called once per video. It may precompute per-frame image
   embeddings, but must not require the entire video in RAM — `frames` is lazy and
   random-access.
3. `add_prompt` must store the prompted frame into the memory bank **and** return
   the mask for that same frame.
4. `propagate(frame_index, direction)` must condition on memory from
   **already-visited frames only**: for `FORWARD`, indices `< frame_index`; for
   `BACKWARD`, indices `> frame_index`. Reading the unvisited side is future
   leakage — it inflates benchmark scores and is caught by
   `api/tests/test_contract.py`.
5. Masks are `(H, W)` `bool` at the `FrameSource` resolution. When
   `object_present` is `False` the mask must be entirely `False`.
6. `propagate` is called at most once per frame per direction, in order, but must
   tolerate being called again for the same frame (the pipeline can retry).
7. `reset()` clears the memory bank and all prompts so the session can restart.

### The methods (exact signatures)

Required (`VideoObjectTracker`, `base.py:253`):

```python
class VideoObjectTracker(abc.ABC):
    key: str = "abstract"                      # registry name, must be unique

    @classmethod
    def info(cls) -> TrackerInfo: ...

    def load(self, *, device: str = "auto", checkpoint: str | Path | None = None) -> None: ...
    def set_video(self, frames: FrameSource) -> None: ...
    def add_prompt(self, prompts: PromptSet) -> FrameResult: ...
    def propagate(self, frame_index: int, direction: Direction) -> FrameResult: ...
```

Optional hooks the app uses when present:

```python
    def reset(self) -> None: ...                       # default: no-op
    def memory_state(self) -> dict[str, Any]: ...      # default: {}
    def warmup(self) -> None: ...                      # default: no-op
```

Training hooks (`TrainableTracker`, `base.py:311`) — implement these if you want
`ml/train.py` to drive the memory stack:

```python
class TrainableTracker(VideoObjectTracker):
    checkpoint_hint: str = ""

    def trainable_parameters(self) -> Iterable[Any]: ...
    def training_step(self, sequence: TrainingSequence) -> dict[str, Any]: ...
```

Payload types (`base.py`):

| Type | Definition |
| --- | --- |
| `Direction` | `FORWARD = "forward"`, `BACKWARD = "backward"` (str enum) |
| `PointPrompt` | `x: float, y: float, positive: bool = True` |
| `BoxPrompt` | `x0, y0, x1, y1: float` (pixel coords of the prompted frame) |
| `PromptSet` | `frame_index: int`, `points: tuple[PointPrompt, ...]`, `box: BoxPrompt \| None` |
| `FrameResult` | `mask: np.ndarray (H, W) bool`, `score: float`, `object_present: bool`, `extras: dict` |
| `TrainingSequence` | `frames (T,H,W,3) uint8`, `gt_masks (T,H,W) bool`, `prompts: tuple[PromptSet, ...]`, `meta: dict` |
| `FrameSource` | `.n_frames`, `.width`, `.height`, `.fps`, `__getitem__(i) -> (H,W,3) uint8 RGB` |

`check_frame_result()` (`base.py:209`) enforces shape, `bool` dtype, `score` in
`[0, 1]`, and `object_present=False => mask.sum() == 0`. It is called on every
result the pipeline receives, so a violation fails the job with your key and frame
index in the message.

---

## 2. Reused vs built — state this precisely

The image encoder is **not** the contribution and is **not** trained here.

| Component | Status |
| --- | --- |
| Hiera image encoder (hierarchical, MAE-pretrained) | **Reused, frozen.** Weights from `facebook/sam2.1-hiera-{tiny,small,base-plus,large}`. Run once per interaction; supplies unconditioned tokens. |
| Prompt encoder (points/boxes -> sparse + dense embeddings) | **Re-implemented** following SAM's design (random Fourier positional encoding; a learned embedding per prompt type). Not a paper contribution. |
| Memory encoder | **Built here.** |
| Memory bank (two queues) | **Built here.** |
| Memory attention | **Built here.** |
| Mask decoder (two-way transformer) | **Built here.** |
| Object pointers | **Built here.** |
| Occlusion / presence head | **Built here.** |

"Built here" means the module is written in this repository and is the thing being
trained. The frozen encoder is a standard, defensible choice: training a
Hiera/ViT backbone from scratch is a different and far more expensive problem, and
it is not what the paper claims.

---

## 3. The memory stack, with the real numbers

Defaults live in the stub's `__init__`: `memory_bank_size=6`,
`num_memory_layers=4`, `embed_dim=256`, `num_heads=8`. Keep those names — the API
and the ablation table in the README quote them.

### 3.1 Memory attention

- `L = 4` transformer blocks.
- Each block, in order: **self-attention -> cross-attention over the memory bank
  and the object pointers -> MLP**. Vanilla multi-head attention, no exotic
  variant.
- **2D spatial RoPE** applied in both self- and cross-attention, on the
  spatial tokens only.
- **Object-pointer tokens are excluded from RoPE** — they have no spatial
  correspondence, so a positional code for them would be meaningless.
- A sinusoidal **absolute** positional embedding is added on top of the RoPE.
- Memory features arrive already projected to a **channel dimension of 64**;
  the attention's internal width is `embed_dim = 256`.

### 3.2 Memory encoder

- Conv-downsample the predicted mask.
- **Sum element-wise with the unconditioned Hiera frame embedding** for that frame.
- Light conv layers fuse the two.
- It does **not** run a second image encoder — that is the whole efficiency
  argument of the design.

### 3.3 Memory bank

- **Two FIFO queues**: up to `N` recent frames, and up to `M` prompted frames.
  `memory_bank_size` (default 6) is the recent-frame budget.
- Both queues store spatial feature maps, not raw frames.
- Temporal position information is embedded into the `N` recent memories (short-term
  motion) and **not** into the prompted memories — the prompted frames are sparser,
  so the signal is harder to generalise.
- Banked features are projected to **dim 64** for cross-attention.
- **Object pointers**: the 256-dim object pointer is split into **4 tokens of 64
  dim** and cross-attended alongside the spatial memory.
- Because the original object pointer is 256-dim, `embed_dim=256` is correct for
  the model even though the bank itself runs at 64.

### 3.4 Mask decoder

- SAM-style **two-way transformer**: token-to-image attention, then image-to-token
  attention, repeated.
- **Skip connections** feed the **stride-4 and stride-8** Hiera features into the
  upsampling path, **bypassing memory attention** entirely. High-resolution detail
  therefore comes straight from the frozen encoder.
- **Multi-mask output** on ambiguous prompts. Only the mask with the **highest
  predicted IoU** is propagated into the memory bank.

### 3.5 Occlusion / presence head

- An **additional output token** alongside the mask tokens and the IoU token.
- An MLP head turns it into a visibility score, reported as
  `FrameResult.object_present`.
- A learned **occlusion embedding** is **also added to the memory features** of
  frames the model predicts occluded, so the bank records "this frame had no
  object" rather than only "here is a mask".

### 3.6 Shape flow

```
frames[i] (H, W, 3) uint8
   -> frozen Hiera  ->  multi-scale features: stride 4 / 8 / 16 / 32
   -> trunk (FpnNeck) -> image_embed  (B, 256, H/16, W/16)
   -> flatten        -> tokens        (B, N, 256)      # N = (H/16)*(W/16)

memory attention(tokens, memory):
        tokens   (B, N, 256)
        memory   (B, M,  64)      # spatial bank, dim 64
        + pointers (B, 4, 64)     # 256-dim pointer split into 4 tokens
        -> (B, N, 256)            # RoPE on spatial tokens; pointers excluded

mask decoder(image_embed, skip4, skip8, tokens_cond, prompt_embeddings)
        -> mask logits (B, K, H/4, W/4)  -> upsample -> (H, W) bool at source res
        -> iou_pred   (B, K)
        -> presence_logit (B,)
```

`N` is the number of spatial tokens at the trunk's stride; `M` is the number of
banked memory positions (recent queue + prompted queue + pointers).

---

## 4. Causality: the rule, and the deliberate deviation

**Our rule.** `propagate` may condition only on frames already visited in that
direction. `FORWARD` reads indices `< frame_index`; `BACKWARD` reads
`> frame_index`. The step planner in `pipeline.py::build_plan` orders prompts and
propagations so this holds across multiple prompts, and the rule is enforced by
`api/tests/test_contract.py` and `ml/rotostream_ml/leakcheck.py`.

**The paper is more permissive.** It states that prompted frames may also come
"from the future" relative to the current frame — its memory design is
**bidirectional** and a prompted frame anywhere in the clip may be attended to.

**This repo is stricter on purpose.** Bidirectional prompting is not reproducible
under a semi-supervised benchmark: it lets a tracker see frames it has not been
asked about yet, which makes a J&F number look better than the deployed behaviour
of a click-then-propagate editor. The directional contract is a **documented
deviation**, not a bug:

- Do **not** "fix" it to match the paper.
- Do **not** adopt bidirectional memory inside the model.
- Do **not** relax the leakage tests.
- The API's `bidirectional` flag is a *user* toggle: it runs a forward sweep and a
  backward sweep from the same prompt, each still causal in its own direction.

---

## 5. Three corrections to the stub's docstrings

The stub shipped with three statements that are wrong, and they mislead anyone
implementing against them.

1. **Losses are FOCAL, not BCE, and there is no temporal-consistency term.**
   The loss is focal (`gamma`-weighted cross-entropy) for the mask, plus Dice, plus
   IoU regression, plus presence cross-entropy. Temporality is *architectural* —
   it comes from memory attention, not from a loss that penalises frame-to-frame
   change. Do not add a temporal-consistency term.

2. **`_object_presence` reads mask-decoder OUTPUT TOKENS, not image features.**
   The additional presence token sits alongside the mask tokens and the IoU token.
   Its parameter was named `features` and the docstring implied image features;
   both are misleading. It consumes the decoder's output tokens.

3. **`embed_dim=256` is the model width, but the memory bank is 64-dim.**
   Banked memory is projected to channel dim **64**, and the 256-dim object pointer
   is split into **4 x 64** tokens for cross-attention. The stub's docstring implied
   the bank runs at 256. A 256-wide bank would be a different (and wrong) memory
   design.

### 5.1 Corrections that only surfaced while implementing

These are not in the stub — they were found by running the thing against the
reference. Each one is easy to reintroduce, and each one wasted real time.

1. **The reference memory path is sequence-first `(seq, batch, ch)`; this stack is
   batch-first `(batch, seq, C)`.** Getting it wrong does not produce a clean
   error. The reference's key-side RoPE repeat computes
   `repeat_factor = k_len // q_len`, so a transposed argument multiplies the
   cos/sin table into a **16 GB** tensor. This crashed the development machine.
   `api/scripts/check_parity.py` transposes explicitly for this reason.
2. **Inference must run under `torch.no_grad()`.** The bank stores features,
   position codes and pointers derived from decoder outputs, so an un-detached
   autograd graph is pinned by the bank and grows with the clip. The symptom is an
   access violation, not a Python exception. `sam2_memory._inference` wraps
   `add_prompt`/`propagate`; only `training_step` builds a graph.
3. **Attention must not be materialised.** With a full bank (6 recent + prompted,
   4096 tokens each) an explicit `q @ k.T` is ~3 GB *per layer* in float32.
   `RoPEAttention` uses `F.scaled_dot_product_attention`, which is both smaller and
   bit-exact against the reference's eager path on CPU float32.
4. **Training needs gradient checkpointing to fit a consumer GPU.** A clip is one
   graph, and the memory encoder's stride-2 pass over a 1024x1024 mask is ~0.25 GB
   of activation per frame. `MemoryStack.set_gradient_checkpointing(True)`
   (enabled by `training_step`) trades ~2x compute for that memory; without it a
   16-frame step sat pinned at the 6 GB ceiling.
5. **`trainable_parameters()` / `state_dict()` / `training_step()` must not
   require `set_video()`.** The harness asks for parameters immediately after
   `load()`. Readiness is split into `_require_loaded()` (weights exist) and
   `_require_ready()` (weights + video; inference only).
6. **`training_step` must return only scalar loggable tensors.** The harness logs
   every value in the returned dict as a scalar metric, so the chosen-mask index
   stays out of `Sam2LossBreakdown.to_dict()`.
7. **The candidate-mask terms reduce over space only.** `sigmoid_focal_loss` and
   `dice_loss` keep their leading dims, so a `(T, K, h, w)` input yields `(T, K)`.
   Reducing over everything collapses the frame axis and the argmin that picks the
   supervised mask then fails.

---

## 6. Losses and training recipe (§D.2.2 of the paper)

| Term | Weight | Notes |
| --- | --- | --- |
| Focal (mask) | 20 | mask prediction |
| Dice (mask) | 1 | |
| IoU regression (MAE) | 1 | predicted IoU; **sigmoid** activation, **L1** loss |
| Object-presence CE | 1 | visibility/occlusion head |

Rules that change the gradient, not just the number:

- Only the mask with the **lowest segmentation loss** is supervised (multi-mask
  output).
- If a frame's ground truth has no mask, **do not supervise the mask outputs**,
  but **always supervise presence** — that is the only signal for "the object is
  gone".
- The paper does **not** add a temporal-consistency loss.

Training recipe from the paper:

- `8`-frame clips.
- Up to `2` frames randomly prompted.
- `7` correction clicks sampled from the GT masklet and from model predictions.
- First prompt sampling: GT mask `p=0.50`, positive click `p=0.25`, box `p=0.25`.
- Augmentation: horizontal flip, resize to `1024x1024`.

### Ablation: memory size (paper Table 9c)

| memories | 4 | 6 | 8 |
| --- | --- | --- | --- |
| J&F | 73.5 | 73.0 | 73.2 |

Essentially flat; **6 is the paper default**, which is why it is the default here.
The README ablation table extends this with the presence head, temporal
positional embedding and object pointers.

### Headline numbers (for context, not targets)

- DAVIS 2017 val: **90.2 J&F** with Hiera-B+ at 43.8 FPS; **90.7** with Hiera-L at
  30.2 FPS.
- Training data: SA-V, 50.9K videos / 35.5M masks.
- Occlusion frequency (Table 3): SA-V manual disappearance rate 42.5% vs DAVIS
  16.1% — the reason the presence head matters more on SA-V than on DAVIS.

---

## 7. Verification order

Run these in order. Each one is cheaper than the next and catches a different
class of bug.

0. **`python api/scripts/check_parity.py`** — the correctness gate, and the reason
   the rest of the list is worth running at all. It loads the released SAM 2.1
   weights into the built memory stack (0 missing / 0 unexpected keys) and compares
   the memory encoder and memory attention against the reference modules on
   identical inputs. A large difference here means the implementation is not the
   architecture, and no benchmark number will save it. Recorded: encoder features,
   encoder position codes and attention output all agree to max abs diff `0.000e+00`.
1. **`python api/scripts/check_model.py sam2_memory`** — contract only: shapes,
   dtype, forward/backward propagation, `reset`, the future-leakage probes, and
   `object_present=False => empty mask`. The fastest loop while iterating.
   Recorded: `8 passed, 0 failed, 0 skipped`.
   (`leakcheck --all` is the slowest of these — it re-runs the tracker twice per
   probe. On CPU, bound it with `--max-checks 3` or it will outlive the session.)
2. **`python -m rotostream_ml.leakcheck --model sam2_memory --all`** (from `ml/`) —
   the strong causality proof: poison every unvisited frame with noise and demand
   the mask at the probe frame is unchanged. Run this before trusting *any* number.
3. **`python -m rotostream_ml.evaluate --model sam2_memory --dataset synthetic`** —
   toy clips with exact ground truth: `linear`, `occlusion`, `reentry`,
   `distractor` (identical lookalike), `color_shift`. Fix directionality and
   off-by-one bugs here, not on real footage.
4. **`python -m rotostream_ml.train --model sam2_memory --dry-run`** then
   **`--overfit`** — verify the training path (non-zero gradients, a loss that can
   fall) before spending GPU hours.
5. **DAVIS**:
   `python -m rotostream_ml.train --model sam2_memory --dataset davis --root ...`
   then
   `python -m rotostream_ml.evaluate --model sam2_memory --dataset davis --root ... --split val --json runs/davis_val.json`.

`check_model.py` prints one line per check. For a stub it reports `SKIP` rather
than a pass — never treat a skip as a green light.

Last full pass, in this order:

| Step | Result |
| --- | --- |
| 0 `check_parity.py` | all three comparisons `0.000e+00`, 0/0 keys |
| 1 `check_model.py sam2_memory` | 8 passed, 0 failed, 0 skipped |
| 2 `leakcheck --all --max-checks 3` | 5 scenarios x 2 directions, all PASS |
| 3 `evaluate --dataset synthetic` | mean J&F 0.9600 over 136 frames |
| 4 `train --dry-run` | PASS, gradients through 305 tensors |
| 4 `train --overfit --steps 50` | PASS: mean loss 0.011446 -> 0.000000 (fell 100%) |

---

## 8. Reference layout (read the architecture, do not copy it)

The upstream repository is the authoritative reading material for shapes and
module boundaries. Read it; do not vendor its files into this repo.

```
github.com/facebookresearch/sam2
  sam2/
    modeling/
      backbones/
        hiera.py             Hiera backbone: stages, windowed attention, multi-scale taps
      image_encoder.py       trunk + FpnNeck; produces the (B, C, H/16, W/16) image embedding
      prompt_encoder.py      Point/Box -> sparse + dense prompt embeddings (SAM's design)
      memory_encoder.py      MaskDownSampler, CXBlock, Fuser  (sec. 3.2)
      memory_attention.py    MemoryAttention, MemoryAttentionLayer, RoPEAttention (sec. 3.1)
      mask_decoder.py        MaskDecoder, TwoWayAttentionBlock, TwoWayTransformer (sec. 3.4)
      position_encoding.py   PositionEmbeddingRandom / Sine
      sam2_base.py           the tracker loop + memory bank bookkeeping (sec. 3.3, 3.5)
      sam2_utils.py          mask/IoU helpers and the focal + dice losses (sec. 6 here)
    configs/sam2.1/
      sam2.1_hiera_t.yaml    the tiny config: copy the *numbers*, not the code
    sam2_video_predictor.py  per-video inference state (how upstream sequences a session)
```

The HF repository `facebook/sam2.1-hiera-tiny` also ships
`sam2.1_hiera_t.yaml`, which is the quickest way to confirm a hyperparameter.

**Use it to answer "what shape does X expect", then write your own module.** The
repo's whole premise is that the memory mechanism is implemented here; copying
upstream defeats it and leaves you unable to defend the code.
