"""Protocol regressions: a good score must not be created by the evaluation setup."""
from __future__ import annotations

import numpy as np
import pytest

from app.models.base import Direction, FrameResult, PromptSet, VideoObjectTracker
from app.models import registry
from rotostream_ml.evaluate import evaluate_tracker
from rotostream_ml.sequences import VideoSequence
from rotostream_ml.davis import Davis2017
from test_davis import make_davis


def sequence(name="clip", objects=1, frames=5):
    masks = np.zeros((frames, objects, 16, 16), dtype=bool)
    masks[:, :, 3:7, 3:7] = True
    return VideoSequence(name=name, masks=masks,
                         frames=np.zeros((frames, 16, 16, 3), dtype=np.uint8))


class MaskEchoTracker(VideoObjectTracker):
    def load(self, *, device="cpu", checkpoint=None):
        pass

    def set_video(self, frames):
        self.source = frames

    def add_prompt(self, prompts):
        assert prompts.mask is not None and not prompts.points
        self.mask = prompts.mask
        return FrameResult(mask=np.zeros_like(self.mask), object_present=False)

    def propagate(self, frame_index, direction):
        # Wrong final frame must not dilute the official score either.
        mask = self.mask if frame_index < self.source.n_frames - 1 else np.zeros_like(self.mask)
        return FrameResult(mask=mask, object_present=bool(mask.any()))


def test_mask_prompts_are_exact_detached_and_read_only():
    clip = sequence()
    expected = clip.masks[0, 0].copy()
    prompt = clip.prompt_for(mode="mask")
    assert not prompt.points and prompt.box is None
    assert np.array_equal(prompt.mask, expected)
    clip.masks[0, 0] = False
    assert np.array_equal(prompt.mask, expected)
    with pytest.raises(ValueError):
        prompt.mask[0, 0] = True


@pytest.mark.parametrize("mask", [np.zeros((3, 3)), np.zeros((3, 3, 1), dtype=bool)])
def test_mask_prompt_requires_boolean_2d_mask(mask):
    with pytest.raises(ValueError, match="boolean array"):
        PromptSet(frame_index=0, mask=mask)


def test_davis_excludes_supplied_mask_and_last_frame_scores_all_objects():
    result = evaluate_tracker(MaskEchoTracker, [sequence(objects=2)], object_index=None,
                              prompt_mode="mask", protocol="davis", progress=False)
    assert result.score.jf == pytest.approx(1)
    assert len(result.sequences) == 2
    assert result.sequences[0].scored_frame_indices == [1, 2, 3]
    assert result.sequences[1].object_index == 1
    assert result.to_dict()["n_sequences"] == 1
    assert result.to_dict()["n_object_tracks"] == 2
    assert result.config["scoring_window"] == "frames_1_through_T_minus_2"
    assert result.csv_rows()[0]["prompt_mode"] == "mask"


@pytest.mark.parametrize("overrides", [dict(prompt_mode="point"), dict(object_index=0),
                                      dict(prompt_frame=2), dict(bidirectional=True),
                                      dict(direction=Direction.BACKWARD)])
def test_davis_rejects_mislabeled_comparisons(overrides):
    options = dict(object_index=None, prompt_mode="mask", protocol="davis", progress=False)
    options.update(overrides)
    with pytest.raises(ValueError, match="DAVIS protocol"):
        evaluate_tracker(MaskEchoTracker, [sequence()], **options)


def test_short_davis_clip_cannot_report_an_empty_perfect_score():
    with pytest.raises(ValueError, match="scoring window"):
        evaluate_tracker(MaskEchoTracker, [sequence(frames=2)], object_index=None,
                         prompt_mode="mask", protocol="davis", progress=False)


def test_diagnostic_only_scores_visited_frames_and_records_real_indices():
    result = evaluate_tracker(MaskEchoTracker, [sequence()], prompt_mode="mask",
                              prompt_frame=2, progress=False)
    assert result.sequences[0].scored_frame_indices == [2, 3, 4]
    assert [frame.index for frame in result.score.sequences[0].frames] == [2, 3, 4]


def test_davis_prefers_2017_split_over_2016(tmp_path):
    make_davis(tmp_path, {"old": 3, "new": 3})
    for year, name in (("2016", "old"), ("2017", "new")):
        folder = tmp_path / "ImageSets" / year
        folder.mkdir(parents=True)
        (folder / "val.txt").write_text(name, encoding="utf-8")
    assert Davis2017(tmp_path, strict=True).names == ["new"]


def test_strict_davis_refuses_to_guess_split(tmp_path):
    make_davis(tmp_path, {"clip": 3})
    with pytest.raises(FileNotFoundError, match="refusing to guess"):
        Davis2017(tmp_path, strict=True)


def test_registry_alias_uses_registered_key(monkeypatch):
    monkeypatch.setenv("ROTOSTREAM_MODELS", "alias=app.models.naive:NaiveColorTracker")
    assert next(info for info in registry.available() if info.name == "alias").implemented


def test_optional_model_reports_missing_dependencies_without_loading(monkeypatch):
    import importlib.util
    from app.models.sam2_memory import MemoryAttentionTracker

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "torch" else real_find_spec(name))
    info = MemoryAttentionTracker.info()
    assert not info.implemented
    assert "torch" in info.error and "requirements-model.txt" in info.error


def test_colour_baseline_can_follow_the_same_mask_prompt_protocol():
    from app.models.naive import NaiveColorTracker
    from rotostream_ml.synthetic import build
    clip = build("linear")
    tracker = NaiveColorTracker()
    tracker.load()
    tracker.set_video(clip.frame_source())
    result = tracker.add_prompt(clip.prompt_for(mode="mask"))
    assert np.array_equal(result.mask, clip.object_masks()[0])
    assert tracker.propagate(1, Direction.FORWARD).mask.any()
