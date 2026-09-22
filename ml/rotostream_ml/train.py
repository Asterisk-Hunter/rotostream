"""Training loop for a ``TrainableTracker`` plugin.

The split of responsibility is deliberate: **you own the model, this owns
everything else.** Optimiser, warmup/cosine schedule, AMP, gradient accumulation
and clipping, checkpointing, resumption, logging, validation and the data
pipeline all live here. Your tracker only has to answer two questions:

* ``trainable_parameters()`` - what should be optimised (usually everything except
  the frozen image encoder).
* ``training_step(sequence)`` - a differentiable forward pass returning
  ``{"loss": tensor, ...}``; every other tensor in the dict is logged.

    # is the wiring correct? three steps, gradient and loss checks, no real training
    python -m rotostream_ml.train --model sam2_memory --dry-run

    # can it actually fit one clip? catches loss/optimiser mistakes in a minute
    python -m rotostream_ml.train --model sam2_memory --overfit --steps 200

    # the real thing, validating with J&F as it goes
    python -m rotostream_ml.train --model sam2_memory --dataset davis --root D:/DAVIS \
        --epochs 20 --eval-every 200 --checkpoint-dir runs/davis

Two design notes worth knowing:

* Validation runs through :func:`rotostream_ml.evaluate.evaluate_tracker` after
  writing a checkpoint to disk, so it exercises ``load(checkpoint=...)`` - the
  same path the API uses. A model that trains but cannot be loaded is caught here
  rather than in the app.
* ``batch_size`` is not a knob. ``training_step`` takes one sequence, so the
  effective batch is ``grad_accum``; pretending otherwise would just be a loop.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from app.models import registry
from app.models.base import PromptSet, TrainableTracker, TrainingSequence, VideoObjectTracker

from .sequences import VideoSequence, interior_point
from .synthetic import available as available_synthetic, build as build_synthetic

__all__ = ["TrainConfig", "TrainReport", "Trainer", "SequenceDataset", "main"]


@dataclass
class TrainConfig:
    """Knobs for the training run. Everything else is fixed policy."""

    model: str = "sam2_memory"
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    max_steps: int | None = None
    grad_accum: int = 1
    clip_grad: float = 1.0
    amp: bool = True
    device: str = "auto"
    checkpoint_dir: Path = Path("runs/train")
    save_every: int = 1
    eval_every: int = 0  # 0 disables validation
    log_every: int = 10
    seed: int = 0
    resume: str | None = None
    #: Frames per training sample. Sequences are sliced into windows this long.
    sequence_length: int = 8
    #: Optional square crop, applied to frames and masks together.
    crop_size: int | None = None
    #: Probability of a horizontal flip per sample.
    flip_prob: float = 0.5
    #: Brightness/contrast jitter strength in [0, 1].
    color_jitter: float = 0.2

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        return payload


@dataclass
class TrainReport:
    """What the run did, so it can be logged or asserted on in a test."""

    steps: int = 0
    epochs: int = 0
    best_val_jf: float | None = None
    final_loss: float | None = None
    first_loss: float | None = None
    history: list[dict] = field(default_factory=list)
    validation: list[dict] = field(default_factory=list)
    checkpoint: str | None = None
    elapsed_s: float = 0.0

    @property
    def loss_decreased(self) -> bool:
        if self.first_loss is None or self.final_loss is None:
            return False
        return self.final_loss < self.first_loss

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


# --------------------------------------------------------------------- dataset
class SequenceDataset:
    """Slices ``VideoSequence`` objects into training samples.

    Augmentation is deliberately applied to frames and masks together: a flip or
    crop that moves the pixels but not the ground truth is the classic silent way
    to train a model that reports great loss and predicts nothing. The prompt is
    rebuilt from the transformed mask afterwards, so it stays on the object.
    """

    def __init__(
        self,
        sequences: Sequence[VideoSequence],
        config: TrainConfig,
        object_index: int = 0,
    ) -> None:
        if not sequences:
            raise ValueError("SequenceDataset needs at least one sequence")
        self.sequences = list(sequences)
        self.config = config
        self.object_index = object_index
        self._rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        """Number of non-overlapping windows available across all sequences."""
        return sum(max(1, seq.n_frames // self.config.sequence_length) for seq in self.sequences)

    def _pick_window(self, sequence: VideoSequence) -> tuple[int, int]:
        length = min(self.config.sequence_length, sequence.n_frames)
        start = int(self._rng.integers(0, max(1, sequence.n_frames - length + 1)))
        return start, start + length

    def _crop(self, frames: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Random square crop, or the identity when the clip is already smaller."""
        size = self.config.crop_size
        if not size:
            return frames, masks
        height, width = frames.shape[1], frames.shape[2]
        size = min(size, height, width)
        top = int(self._rng.integers(0, height - size + 1))
        left = int(self._rng.integers(0, width - size + 1))
        return (
            frames[:, top : top + size, left : left + size],
            masks[:, top : top + size, left : left + size],
        )

    def _photometric(self, frames: np.ndarray) -> np.ndarray:
        strength = self.config.color_jitter
        if strength <= 0:
            return frames
        gain = 1.0 + float(self._rng.uniform(-strength, strength))
        bias = float(self._rng.uniform(-strength, strength)) * 40.0
        out = frames.astype(np.float32) * gain + bias
        return np.clip(out, 0, 255).astype(np.uint8)

    def sample(self, index: int | None = None) -> TrainingSequence:
        """Build one augmented training sample."""
        sequence = self.sequences[int(self._rng.integers(0, len(self.sequences)))]
        start, stop = self._pick_window(sequence)
        frames = np.asarray(sequence.frames[start:stop]) if sequence.frames is not None else None
        if frames is None:
            source = sequence.frame_source()
            frames = np.stack([source[i] for i in range(start, stop)])
        masks = sequence.object_masks(self.object_index)[start:stop]

        frames, masks = self._crop(frames, masks)
        if self.config.flip_prob > 0 and self._rng.random() < self.config.flip_prob:
            frames = frames[:, :, ::-1]
            masks = masks[:, :, ::-1]
        frames = np.ascontiguousarray(frames)
        masks = np.ascontiguousarray(masks)
        frames = self._photometric(frames)

        prompts = self._prompts_for(masks)
        return TrainingSequence(
            frames=frames,
            gt_masks=masks,
            prompts=prompts,
            meta={"sequence": sequence.name, "start": start, "stop": stop},
        )

    def _prompts_for(self, masks: np.ndarray) -> tuple[PromptSet, ...]:
        """Prompt on the first frame in the window where the object is visible."""
        visible = masks.reshape(masks.shape[0], -1).any(axis=1)
        indices = np.nonzero(visible)[0]
        if len(indices) == 0:
            # An all-empty window is legitimately possible (the object is off
            # screen throughout); the tracker is then given no prompt and is
            # expected to learn "nothing here".
            return ()
        index = int(indices[0])
        point = interior_point(masks[index])
        if point is None:  # pragma: no cover - guarded by `visible` above
            return ()
        from app.models.base import PointPrompt

        return (
            PromptSet(
                frame_index=index,
                points=(PointPrompt(x=point[0], y=point[1], positive=True),),
            ),
        )

    def epoch_iter(self, steps: int) -> Iterator[TrainingSequence]:
        for _ in range(steps):
            yield self.sample()


# ------------------------------------------------------------------- checkpoint
def capture_state(tracker: TrainableTracker) -> dict:
    """Snapshot trainable weights in a way that does not require new API.

    A tracker that defines ``state_dict``/``load_state_dict`` (the usual torch
    convention) is used directly. Otherwise the tensors from
    ``trainable_parameters()`` are saved in order, which is why ``restore_state``
    must see the same construction order - another reason to provide a
    ``state_dict`` if you can.
    """
    if hasattr(tracker, "state_dict"):
        return {
            "kind": "state_dict",
            "state": {key: value.detach().cpu() for key, value in tracker.state_dict().items()},
        }
    return {
        "kind": "params",
        "state": [param.detach().cpu().clone() for param in tracker.trainable_parameters()],
    }


def restore_state(tracker: TrainableTracker, blob: dict) -> None:
    """Load a snapshot produced by :func:`capture_state`."""
    import torch

    if blob["kind"] == "state_dict":
        if not hasattr(tracker, "load_state_dict"):
            raise RuntimeError("checkpoint holds a state_dict but the tracker cannot load one")
        tracker.load_state_dict({k: v for k, v in blob["state"].items()})
        return
    params = list(tracker.trainable_parameters())
    saved = blob["state"]
    if len(params) != len(saved):
        raise RuntimeError(
            f"checkpoint has {len(saved)} tensors but the tracker exposes {len(params)}; "
            "the model structure changed since it was written"
        )
    with torch.no_grad():
        for param, value in zip(params, saved):
            param.copy_(value.to(param.device))


# ----------------------------------------------------------------------- trainer
class Trainer:
    """Drives ``TrainableTracker.training_step`` with all the usual machinery."""

    def __init__(
        self,
        tracker: TrainableTracker,
        dataset: SequenceDataset,
        config: TrainConfig,
        validation_sequences: Sequence[VideoSequence] | None = None,
    ) -> None:
        import torch

        if not isinstance(tracker, TrainableTracker):
            raise TypeError(
                f"{type(tracker).__name__} does not implement TrainableTracker; "
                "add trainable_parameters() and training_step() to train it"
            )

        self.torch = torch
        self.tracker = tracker
        self.dataset = dataset
        self.config = config
        self.validation_sequences = list(validation_sequences or [])

        self.device = _resolve_device(config.device)
        # The optimiser is built in load_tracker(), not here: parameters only exist
        # after the model is constructed, and constructing it is exactly what
        # load() does. Building the optimiser first would bind None.
        self.params: list[Any] = []
        self.optimizer = None
        self.scaler = None
        self.report = TrainReport()
        self._step = 0
        self._epoch = 0

    # ------------------------------------------------------------------ setup
    def _amp_enabled(self) -> bool:
        return bool(self.config.amp and self.device.startswith("cuda"))

    def total_steps(self) -> int:
        if self.config.max_steps is not None:
            return int(self.config.max_steps)
        per_epoch = max(1, len(self.dataset))
        return int(per_epoch * self.config.epochs)

    def _scheduler(self):
        total = max(1, self.total_steps())
        warmup = min(self.config.warmup_steps, max(1, total // 10))

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return self.torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def load_tracker(self, checkpoint: str | None = None) -> None:
        """Bring up the model, then wire the optimiser to its parameters.

        Order matters and is the reason ``load_tracker`` is not optional: a model
        that creates its parameters inside ``load`` has nothing to optimise until
        it has been called.
        """
        self.tracker.load(device=self.device, checkpoint=checkpoint)
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        torch = self.torch
        self.params = [
            param for param in self.tracker.trainable_parameters() if param.requires_grad
        ]
        if not self.params:
            raise RuntimeError(
                f"{self.config.model} exposes no trainable parameters. If the image "
                "encoder is frozen, make sure the memory stack is still returned from "
                "trainable_parameters()."
            )
        self.optimizer = torch.optim.AdamW(
            self.params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled())

    def _require_optimizer(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("call load_tracker() before training or stepping the optimiser")

    # ------------------------------------------------------------------- steps
    def train_step(self, sequence: TrainingSequence) -> dict[str, float]:
        """One optimiser step, accumulating ``grad_accum`` samples of gradient."""
        torch = self.torch
        self._require_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(max(1, self.config.grad_accum)):
            sample = sequence if self.config.grad_accum == 1 else self.dataset.sample()
            with torch.autocast(
                device_type="cuda" if self.device.startswith("cuda") else "cpu",
                enabled=self._amp_enabled(),
            ):
                output = self.tracker.training_step(sample)
            loss = output["loss"]
            if not torch.is_tensor(loss):
                raise TypeError(
                    f"training_step must return tensors; got {type(loss).__name__} for 'loss'"
                )
            scaled = loss / max(1, self.config.grad_accum)
            self.scaler.scale(scaled).backward()

            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    accumulated[key] = accumulated.get(key, 0.0) + float(value.detach().mean())
            accumulated["loss"] = accumulated.get("loss", 0.0) + float(loss.detach())

        self.scaler.unscale_(self.optimizer)
        if self.config.clip_grad > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.config.clip_grad)
            accumulated["grad_norm"] = float(grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return accumulated

    def validate(self) -> dict[str, float]:
        """Write a checkpoint and score it through the same path the API uses."""
        from .evaluate import evaluate_tracker

        self._require_optimizer()

        path = self.save(tag="latest")
        evaluation = evaluate_tracker(
            self.config.model,
            self.validation_sequences,
            device=self.device,
            dataset="validation",
            checkpoint=str(path),
            progress=False,
        )
        record = {
            "step": self._step,
            "j": evaluation.score.j,
            "f": evaluation.score.f,
            "jf": evaluation.score.jf,
            "presence_accuracy": evaluation.presence.accuracy,
        }
        self.report.validation.append(record)
        if self.report.best_val_jf is None or record["jf"] > self.report.best_val_jf:
            self.report.best_val_jf = record["jf"]
            self.save(tag="best")
        return record

    # ------------------------------------------------------------- persistence
    def save(self, tag: str = "latest") -> Path:
        """Checkpoint weights + optimiser + step, so a run can resume exactly."""
        self._require_optimizer()
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{tag}.pt"

        state = {
            "model": self.config.model,
            "step": self._step,
            "epoch": self._epoch,
            "config": self.config.to_dict(),
            "weights": capture_state(self.tracker),
            "optimizer": self.optimizer.state_dict(),
            "torch_rng": self.torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        self.torch.save(state, path)
        self.report.checkpoint = str(path)
        return path

    def resume(self, path: str | Path) -> None:
        """Restore a run. Called after ``load_tracker`` so weights exist."""
        self._require_optimizer()
        blob = self.torch.load(Path(path), map_location="cpu", weights_only=False)
        restore_state(self.tracker, blob["weights"])
        try:
            self.optimizer.load_state_dict(blob["optimizer"])
        except (ValueError, KeyError) as exc:  # structure changed since the checkpoint
            print(f"  warning: could not restore the optimiser state ({exc}); starting fresh")
        self._step = int(blob.get("step", 0))
        self._epoch = int(blob.get("epoch", 0))
        self.torch.set_rng_state(blob["torch_rng"])
        np.random.set_state(blob["numpy_rng"])
        random.setstate(blob["python_rng"])
        print(f"resumed {path} at step {self._step}")

    # -------------------------------------------------------------------- loop
    def fit(self) -> TrainReport:
        self._require_optimizer()
        total = self.total_steps()
        scheduler = self._scheduler()
        started = time.perf_counter()

        log_path = Path(self.config.checkpoint_dir) / "metrics.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        while self._step < total:
            self._epoch += 1
            for sample in self.dataset.epoch_iter(max(1, len(self.dataset))):
                if self._step >= total:
                    break
                metrics = self.train_step(sample)
                scheduler.step()
                self._step += 1

                if self.report.first_loss is None:
                    self.report.first_loss = metrics["loss"]
                self.report.final_loss = metrics["loss"]
                record = {"step": self._step, "epoch": self._epoch, **metrics}
                record["lr"] = self.optimizer.param_groups[0]["lr"]
                self.report.history.append(record)

                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

                if self.config.log_every and self._step % self.config.log_every == 0:
                    extras = "  ".join(
                        f"{key} {value:.4f}"
                        for key, value in metrics.items()
                        if key not in {"loss", "grad_norm"}
                    )
                    print(
                        f"  step {self._step:6d}/{total}  loss {metrics['loss']:.4f}  "
                        f"lr {record['lr']:.2e}  {extras}",
                        flush=True,
                    )

                if self.config.eval_every and self._step % self.config.eval_every == 0:
                    if self.validation_sequences:
                        val = self.validate()
                        print(
                            f"  eval  step {self._step:6d}  J {val['j']:.4f}  F {val['f']:.4f}  "
                            f"J&F {val['jf']:.4f}",
                            flush=True,
                        )
                    self.save()

            if self.config.save_every and self._epoch % self.config.save_every == 0:
                self.save()

        self.report.steps = self._step
        self.report.epochs = self._epoch
        self.report.elapsed_s = time.perf_counter() - started
        # Final checkpoint always: a crash-free run should still leave weights.
        self.report.checkpoint = str(self.save())
        return self.report


def _resolve_device(requested: str) -> str:
    from app.models.base import resolve_device

    return resolve_device(requested)


# ----------------------------------------------------------------------- modes
def dry_run(tracker: TrainableTracker, dataset: SequenceDataset, config: TrainConfig) -> int:
    """Three steps with hard checks - verifies the wiring, not the science.

    Catches the failure modes that waste a day: no trainable parameters, a loss
    with no gradient (a detached tensor), gradients that are all zero (a
    disconnected branch), and a loss that cannot move at all.
    """
    import torch

    trainer = Trainer(tracker, dataset, config)
    trainer.load_tracker()
    print(f"dry run: {config.model} on {len(dataset)} sample(s), device={trainer.device}")
    print(f"  trainable tensors: {len(trainer.params)}")

    sample = dataset.sample()
    print(
        f"  sample: frames {sample.frames.shape} gt {sample.gt_masks.shape} "
        f"prompts {len(sample.prompts)}"
    )

    losses = []
    grad_norms = []
    for _ in range(3):
        trainer.optimizer.zero_grad(set_to_none=True)
        output = tracker.training_step(sample)
        loss = output["loss"]
        if not torch.is_tensor(loss):
            print(f"\nFAIL: 'loss' is {type(loss).__name__}, not a tensor")
            return 1
        if not loss.requires_grad:
            print("\nFAIL: loss does not require grad - the graph is detached somewhere")
            return 1
        loss.backward()
        norms = {
            name: float(param.grad.norm())
            for name, param in zip(_param_names(tracker), trainer.params)
            if param.grad is not None
        }
        with_grad = {name: norm for name, norm in norms.items() if norm > 0}
        losses.append(float(loss.detach()))
        grad_norms.append(float(sum(norms.values())))
        print(
            f"  loss {losses[-1]:.4f}  |grad| {grad_norms[-1]:.4f}  "
            f"params with non-zero grad: {len(with_grad)}/{len(trainer.params)}"
        )
        if not with_grad:
            print(
                "\nFAIL: every gradient is zero. training_step is probably not using the "
                "prompted features, or the trainable modules are not on the compute path."
            )
            return 1

    if len(set(round(value, 6) for value in losses)) == 1 and grad_norms[0] == 0:
        print("\nFAIL: the loss never changed and no gradient flowed")
        return 1

    print(
        "\nPASS: the training path is wired up. "
        f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
        f"gradients flowing through {len(trainer.params)} tensors."
    )
    return 0


def _param_names(tracker: TrainableTracker) -> list[str]:
    """Best-effort parameter names, for readable dry-run output."""
    if hasattr(tracker, "named_parameters"):
        return [name for name, _ in tracker.named_parameters()]  # type: ignore[attr-defined]
    return [f"param[{index}]" for index in range(len(list(tracker.trainable_parameters())))]


def overfit(tracker: TrainableTracker, sequences: Sequence[VideoSequence], config: TrainConfig) -> int:
    """Can the model fit a single clip? The fastest real signal that it can learn.

    Reports the loss trend and fails if it did not fall, because "the loss went
    down on one sequence" is the cheapest possible evidence that the memory stack
    and its loss are connected correctly.
    """
    import torch

    config.sequence_length = max(config.sequence_length, min(16, sequences[0].n_frames))
    config.epochs = 1
    dataset = SequenceDataset(sequences[:1], config)
    trainer = Trainer(tracker, dataset, config)
    trainer.load_tracker()

    print(f"overfit: {sequences[0].name} x {config.max_steps} steps, lr {config.lr}")
    window = dataset.sample()
    history = []
    for step in range(1, config.max_steps + 1):
        metrics = trainer.train_step(window)
        history.append(metrics["loss"])
        if step == 1 or step % max(1, config.max_steps // 10) == 0:
            print(f"  step {step:5d}  loss {metrics['loss']:.6f}")

    first = float(np.mean(history[: max(1, len(history) // 10)]))
    last = float(np.mean(history[-max(1, len(history) // 10) :]))
    print(f"\n  mean loss first 10%: {first:.6f}")
    print(f"  mean loss last 10%:  {last:.6f}")
    if last >= first:
        print("\nFAIL: the loss did not decrease. Something is wrong upstream of the optimiser.")
        return 1
    print(f"\nPASS: loss fell {first - last:.6f} ({100 * (first - last) / first:.1f}%).")
    return 0


# ------------------------------------------------------------------------- cli
def _build_sequences(args: argparse.Namespace) -> tuple[list[VideoSequence], list[VideoSequence]]:
    """Training and validation sequences, matched to the chosen dataset."""
    if args.dataset == "synthetic":
        names = args.sequences or available_synthetic()
        sequences = [build_synthetic(name) for name in names]
        return sequences, sequences

    if args.dataset == "davis":
        from .davis import Davis2017

        if not args.root:
            raise SystemExit("--root is required for --dataset davis")
        dataset = Davis2017(root=args.root, split=args.split, resolution=args.resolution)
        names = dataset.names
        if args.limit:
            names = names[: args.limit]
        train_names = names[: max(1, int(len(names) * 0.8))]
        val_names = names[len(train_names) :] or train_names[:1]
        print(f"DAVIS {args.split}: {len(train_names)} train / {len(val_names)} val sequences")
        return [dataset.load(name) for name in train_names], [dataset.load(name) for name in val_names]

    raise SystemExit(f"unknown dataset {args.dataset!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a TrainableTracker plugin.")
    parser.add_argument("--model", default="sam2_memory", help="registered tracker key")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "davis"])
    parser.add_argument("--sequences", nargs="*", default=None, help="synthetic scenario names")
    parser.add_argument("--root", default=None, help="DAVIS root directory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None, help="max optimiser steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None, help="initial weights for the tracker")
    parser.add_argument("--resume", default=None, help="resume a training checkpoint")
    parser.add_argument("--checkpoint-dir", default="runs/train")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="check the wiring in three steps")
    parser.add_argument("--overfit", action="store_true", help="fit one clip and verify the loss falls")
    args = parser.parse_args(argv)

    try:
        tracker = registry.create(args.model)
    except KeyError as exc:
        print(exc)
        return 1

    if not isinstance(tracker, TrainableTracker):
        print(
            f"{args.model} is a VideoObjectTracker but not a TrainableTracker.\n"
            "Add trainable_parameters() and training_step() to train it (see "
            "api/app/models/base.py)."
        )
        return 1

    config = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps,
        grad_accum=args.grad_accum,
        amp=not args.no_amp,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        eval_every=args.eval_every,
        log_every=args.log_every,
        seed=args.seed,
        resume=args.resume,
        sequence_length=args.sequence_length,
        crop_size=args.crop_size,
    )

    train_sequences, validation_sequences = _build_sequences(args)

    try:
        if args.dry_run:
            return dry_run(tracker, SequenceDataset(train_sequences, config), config)
        if args.overfit:
            config.max_steps = config.max_steps or 50
            return overfit(tracker, train_sequences, config)

        trainer = Trainer(
            tracker,
            SequenceDataset(train_sequences, config),
            config,
            validation_sequences=validation_sequences if args.eval_every else None,
        )
        trainer.load_tracker(checkpoint=args.checkpoint)
        if args.resume:
            trainer.resume(args.resume)
        report = trainer.fit()
    except NotImplementedError as exc:
        print(f"\n{args.model} is registered but not implemented yet:\n  {exc}\n")
        return 2
    except FileNotFoundError as exc:
        print(f"\n{args.model} needs a checkpoint:\n  {exc}\n")
        return 3

    print(
        f"\nfinished {report.steps} steps in {report.elapsed_s:.1f}s  "
        f"loss {report.first_loss:.4f} -> {report.final_loss:.4f}"
        if report.first_loss is not None
        else f"\nfinished {report.steps} steps in {report.elapsed_s:.1f}s"
    )
    if report.checkpoint:
        print(f"checkpoint: {report.checkpoint}")
    if report.best_val_jf is not None:
        print(f"best J&F: {report.best_val_jf:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
