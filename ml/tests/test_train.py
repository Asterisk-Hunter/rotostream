"""Training-loop tests, driven by a tiny tracker that can actually learn.

The loop is the part of the harness you cannot eyeball: an optimiser that never
steps, gradients that never flow and an off-by-one in the augmentation all look
like "the loss is not moving". So the tests use a real (if trivial) trainable
model and assert the things that would otherwise waste a GPU-day.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.models.base import (  # noqa: E402
    Direction,
    FrameResult,
    PromptSet,
    TrackerInfo,
    TrainableTracker,
    TrainingSequence,
    VideoObjectTracker,
)
from rotostream_ml.synthetic import OBJECT_COLOURS, build  # noqa: E402
from rotostream_ml.train import (  # noqa: E402
    SequenceDataset,
    TrainConfig,
    Trainer,
    capture_state,
    dry_run,
    overfit,
    restore_state,
)


class TinyThresholdTracker(TrainableTracker):
    """Learns a threshold on "redness": about the smallest thing that can train.

    Deliberately has a learnable parameter whose gradient depends on the input, so
    a broken loop shows up as a flat loss rather than passing by accident.
    """

    key = "tiny_threshold"
    checkpoint_hint = "none"

    def __init__(self) -> None:
        self.weight = None
        self.bias = None
        self.device = "cpu"
        self._frames = None

    @classmethod
    def info(cls) -> TrackerInfo:
        return TrackerInfo(
            name=cls.key, description="tiny trainable tracker", trainable=True, uses_memory=True
        )

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        from app.models.base import resolve_device

        self.device = resolve_device(device)
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        if checkpoint:
            blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
            restore_state(self, blob["weights"])

    def set_video(self, frames) -> None:  # noqa: ANN001
        self._frames = frames

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        return FrameResult(mask=np.zeros(self._frames.shape, dtype=bool))

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        return FrameResult(mask=np.zeros(self._frames.shape, dtype=bool))

    def trainable_parameters(self):
        return [self.weight, self.bias]

    def training_step(self, sequence: TrainingSequence) -> dict:
        frames = torch.from_numpy(sequence.frames).float().permute(0, 3, 1, 2) / 255.0
        ground_truth = torch.from_numpy(sequence.gt_masks).float().unsqueeze(1)
        redness = frames[:, 0:1] - frames[:, 1:2]  # the object is the red thing
        logits = self.weight * (redness - self.bias)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ground_truth)
        return {"loss": loss, "mean_logit": logits.mean()}


class FrozenTracker(TrainableTracker):
    """Trainable in name only: no parameters, so the trainer must refuse it."""

    key = "frozen"

    def load(self, *, device="auto", checkpoint=None) -> None:  # noqa: ANN001
        pass

    def set_video(self, frames) -> None:  # noqa: ANN001
        pass

    def add_prompt(self, prompts: PromptSet) -> FrameResult:
        raise NotImplementedError

    def propagate(self, frame_index: int, direction: Direction) -> FrameResult:
        raise NotImplementedError

    def trainable_parameters(self):
        return []

    def training_step(self, sequence: TrainingSequence) -> dict:
        return {"loss": torch.tensor(1.0, requires_grad=True)}


def _config(**overrides) -> TrainConfig:
    defaults = dict(
        model="tiny_threshold",
        lr=0.05,
        epochs=1,
        max_steps=40,
        warmup_steps=5,
        grad_accum=1,
        amp=False,
        device="cpu",
        log_every=0,
        seed=0,
        sequence_length=6,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _sequences():
    return [build("linear"), build("occlusion")]


# ------------------------------------------------------------------- dataset
def test_samples_have_the_shapes_the_contract_promises():
    dataset = SequenceDataset(_sequences(), _config())
    sample = dataset.sample()

    assert sample.frames.ndim == 4 and sample.frames.shape[-1] == 3
    assert sample.frames.dtype == np.uint8
    assert sample.gt_masks.ndim == 3
    assert sample.frames.shape[:3] == sample.gt_masks.shape
    assert sample.frames.shape[0] <= 6


@pytest.mark.parametrize(
    "crop,flip",
    [(None, 0.0), (48, 0.0), (None, 1.0), (48, 1.0)],
    ids=["no-aug", "crop", "flip", "crop+flip"],
)
def test_augmentation_keeps_frames_and_masks_aligned(crop, flip):
    """A flip or crop that moves pixels but not ground truth is a silent disaster.

    The object has a unique colour, so "the mask is exactly the painted region" is
    a complete statement about alignment: no off-by-one in either axis, and no
    flip applied to one of the two arrays only.
    """
    colour = np.array(OBJECT_COLOURS[0], dtype=np.uint8)
    dataset = SequenceDataset(
        [build("linear", size=24)],
        _config(crop_size=crop, flip_prob=flip, color_jitter=0.0),
    )

    for _ in range(8):
        sample = dataset.sample()
        for index, (frame, mask) in enumerate(zip(sample.frames, sample.gt_masks)):
            painted = np.all(frame == colour, axis=-1)
            assert np.array_equal(mask, painted), (
                f"frame {index}: mask and object pixels disagree "
                f"(mask {mask.sum()}px, painted {painted.sum()}px)"
            )


def test_prompts_survive_augmentation_and_stay_on_the_object():
    dataset = SequenceDataset(
        [build("linear")], _config(crop_size=64, flip_prob=1.0, color_jitter=0.0)
    )
    for _ in range(8):
        sample = dataset.sample()
        assert sample.prompts, "an augmented sample should still be prompted"
        prompt = sample.prompts[0]
        point = prompt.points[0]
        assert sample.gt_masks[prompt.frame_index][int(point.y), int(point.x)]


def test_photometric_jitter_changes_pixels_but_not_masks():
    dataset = SequenceDataset([build("linear")], _config(color_jitter=0.5, flip_prob=0.0))
    sample = dataset.sample()
    reference = dataset._photometric(sample.frames)
    assert reference.shape == sample.frames.shape
    assert reference.dtype == np.uint8


def test_dataset_length_counts_windows():
    dataset = SequenceDataset(_sequences(), _config(sequence_length=6))
    assert len(dataset) > 0


def test_dataset_requires_at_least_one_sequence():
    with pytest.raises(ValueError, match="at least one sequence"):
        SequenceDataset([], _config())


# ------------------------------------------------------------------ dry run
def test_dry_run_passes_for_a_working_model():
    tracker = TinyThresholdTracker()
    exit_code = dry_run(tracker, SequenceDataset(_sequences(), _config()), _config())
    assert exit_code == 0


def test_dry_run_defers_to_the_tracker_when_it_is_not_implemented():
    from app.models.sam2_memory import MemoryAttentionTracker

    with pytest.raises(NotImplementedError):
        dry_run(MemoryAttentionTracker(), SequenceDataset(_sequences(), _config()), _config())


# ------------------------------------------------------------------ overfit
def test_overfit_drives_the_loss_down():
    sequence = build("linear")
    exit_code = overfit(TinyThresholdTracker(), [sequence], _config(max_steps=60, lr=0.05))
    assert exit_code == 0, "the loss should fall on a single clip"


# -------------------------------------------------------------------- trainer
def test_trainer_refuses_a_tracker_with_no_trainable_parameters():
    # The check happens when the optimiser is built, i.e. after load(): parameters
    # do not exist before the model does.
    trainer = Trainer(FrozenTracker(), SequenceDataset(_sequences(), _config()), _config())
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        trainer.load_tracker()


def test_using_the_trainer_before_loading_says_so():
    trainer = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), _config()), _config())
    with pytest.raises(RuntimeError, match="load_tracker"):
        trainer.fit()
    with pytest.raises(RuntimeError, match="load_tracker"):
        trainer.save()


def test_trainer_refuses_a_plain_tracker():
    from app.models.naive import NaiveColorTracker

    with pytest.raises(TypeError, match="TrainableTracker"):
        Trainer(NaiveColorTracker(), SequenceDataset(_sequences(), _config()), _config())


def test_training_records_history_and_lowers_the_loss(tmp_path):
    config = _config(max_steps=40, checkpoint_dir=tmp_path)
    trainer = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), config), config)
    trainer.load_tracker()
    report = trainer.fit()

    assert report.steps == 40
    assert len(report.history) == 40
    assert report.loss_decreased, f"loss {report.first_loss} -> {report.final_loss}"
    assert (tmp_path / "metrics.jsonl").is_file()
    assert (tmp_path / "latest.pt").is_file()


def test_gradients_actually_flow_into_the_parameters():
    config = _config()
    trainer = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), config), config)
    trainer.load_tracker()
    before = [param.detach().clone() for param in trainer.params]

    trainer.train_step(trainer.dataset.sample())

    moved = [
        not torch.equal(before[index], param.detach())
        for index, param in enumerate(trainer.params)
    ]
    assert any(moved), "no parameter changed after an optimiser step"


def test_learning_rate_schedule_warms_up_and_decays():
    config = _config(max_steps=100, warmup_steps=10)
    trainer = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), config), config)
    trainer.load_tracker()
    scheduler = trainer._scheduler()

    peaks = []
    for _ in range(100):
        peaks.append(trainer.optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert peaks[0] < peaks[9], "warmup should rise"
    assert peaks[-1] < peaks[50], "cosine should decay"


# --------------------------------------------------------------- persistence
def test_state_capture_and_restore_round_trip():
    tracker = TinyThresholdTracker()
    tracker.load(device="cpu")
    with torch.no_grad():
        tracker.weight.fill_(3.0)
        tracker.bias.fill_(-1.5)

    blob = capture_state(tracker)
    with torch.no_grad():
        tracker.weight.zero_()
        tracker.bias.zero_()
    restore_state(tracker, blob)

    assert float(tracker.weight) == pytest.approx(3.0)
    assert float(tracker.bias) == pytest.approx(-1.5)


def test_checkpoint_round_trips_through_a_fresh_tracker(tmp_path):
    config = _config(max_steps=20, checkpoint_dir=tmp_path)
    trainer = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), config), config)
    trainer.load_tracker()
    trainer.fit()

    path = trainer.save(tag="snapshot")
    assert path.is_file()

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert set(blob) >= {"model", "step", "config", "weights", "optimizer"}

    fresh = TinyThresholdTracker()
    fresh.load(device="cpu")
    restore_state(fresh, blob["weights"])
    for original, restored in zip(trainer.params, fresh.trainable_parameters()):
        assert torch.allclose(original.detach(), restored.detach())


def test_resuming_restores_the_step_counter(tmp_path):
    config = _config(max_steps=10, checkpoint_dir=tmp_path)
    trainer = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), config), config)
    trainer.load_tracker()
    trainer.fit()
    path = trainer.save()

    resumed = Trainer(TinyThresholdTracker(), SequenceDataset(_sequences(), config), config)
    resumed.load_tracker()
    resumed.resume(path)
    assert resumed._step == trainer._step


def test_restoring_into_a_differently_shaped_model_fails_loudly():
    tracker = TinyThresholdTracker()
    tracker.load(device="cpu")
    blob = capture_state(tracker)
    blob["state"] = blob["state"][:1]  # pretend the architecture changed

    with pytest.raises(RuntimeError, match="checkpoint"):
        restore_state(tracker, blob)


def test_validation_writes_a_checkpoint_and_reports_jf(tmp_path, monkeypatch):
    """Validation must go through the on-disk checkpoint, like the app does."""
    from rotostream_ml import evaluate as evaluate_module

    captured: dict = {}

    class FakeScore:
        j, f, jf = 0.5, 0.6, 0.55

    class FakePresence:
        accuracy = 0.9

    class FakeEvaluation:
        score = FakeScore()
        presence = FakePresence()

    def fake_evaluate(model, sequences, **kwargs):
        captured["checkpoint"] = kwargs.get("checkpoint")
        captured["model"] = model
        return FakeEvaluation()

    monkeypatch.setattr(evaluate_module, "evaluate_tracker", fake_evaluate)

    config = _config(max_steps=1, checkpoint_dir=tmp_path)
    trainer = Trainer(
        TinyThresholdTracker(),
        SequenceDataset(_sequences(), config),
        config,
        validation_sequences=_sequences(),
    )
    trainer.load_tracker()
    record = trainer.validate()

    assert record["jf"] == pytest.approx(0.55)
    assert captured["checkpoint"] is not None
    assert captured["checkpoint"].endswith("latest.pt")
    assert trainer.report.best_val_jf == pytest.approx(0.55)


# ------------------------------------------------------------------- config
def test_config_serialises_paths_for_the_checkpoint():
    payload = _config(checkpoint_dir="runs/x").to_dict()
    assert payload["checkpoint_dir"] == "runs/x"
    assert payload["model"] == "tiny_threshold"
