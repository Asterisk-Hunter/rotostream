"""The review rules: what the API is allowed to say about a finished run."""
from __future__ import annotations

from app.quality import session_quality, summarize_quality
from app.video import build_manifest


def scores_for(n_frames: int, absent: list[int], low: list[int], prompted: list[int]) -> list[dict]:
    return [
        {
            "frame_index": index,
            "score": 0.2 if index in low else 0.9,
            "object_present": index not in absent,
            "prompted": index in prompted,
        }
        for index in range(n_frames)
    ]


def test_missing_frames_and_weak_frames_are_kept_apart() -> None:
    summary = summarize_quality(
        scores_for(20, absent=[3, 4], low=[10, 11], prompted=[0]),
        n_frames=20,
        prompt_frames=[0],
    )
    assert summary["absent_frames"] == [3, 4]
    assert summary["low_confidence_frames"] == [10, 11]
    assert summary["problem_frames"] == [3, 4, 10, 11]
    assert summary["n_masked"] == 18
    assert summary["coverage"] == 0.9
    assert summary["sound"] is False


def test_a_prompted_frame_is_never_reported_as_low_confidence() -> None:
    """The user's own click is ground truth; a low score there is not a mask defect."""
    summary = summarize_quality(
        scores_for(10, absent=[], low=[2], prompted=[2]),
        n_frames=10,
        prompt_frames=[2],
    )
    assert summary["low_confidence_frames"] == []
    assert summary["sound"] is True


def test_a_frame_marked_as_background_is_not_lost_tracking() -> None:
    summary = summarize_quality(
        scores_for(10, absent=[6], low=[], prompted=[6]),
        n_frames=10,
        prompt_frames=[6],
        background_only_frames=[6],
    )
    assert summary["absent_frames"] == []
    assert summary["background_only_frames"] == [6]
    assert summary["sound"] is True


def test_an_old_session_is_described_from_its_own_scores() -> None:
    """No stored summary must mean "derive it", never "assume it went fine"."""
    summary = session_quality(
        {
            "n_frames": 12,
            "prompt_frames": [0],
            "absent_frames": [4, 5, 6],
            "scores": scores_for(12, absent=[4, 5, 6], low=[], prompted=[0]),
        }
    )
    assert summary["n_masked"] == 9
    assert summary["coverage"] == 0.75
    assert summary["sound"] is False
    assert summary["problem_frames"] == [4, 5, 6]


def test_a_recorded_absent_list_wins_over_a_re_derivation() -> None:
    """A 2026 run recorded its own absent frames; that list saw more than the meta kept."""
    summary = session_quality(
        {
            "n_frames": 12,
            "prompt_frames": [0],
            "absent_frames": [2, 3],
            "scores": scores_for(12, absent=[3], low=[], prompted=[0]),
        }
    )
    assert summary["absent_frames"] == [2, 3]
    assert summary["n_masked"] == 10
    assert summary["problem_frames"] == [2, 3]
    assert summary["sound"] is False


def test_a_stored_summary_is_returned_untouched() -> None:
    stored = summarize_quality(scores_for(5, absent=[], low=[], prompted=[0]), n_frames=5, prompt_frames=[0])
    assert session_quality({"quality": stored, "n_frames": 5, "scores": []}) == stored


def test_an_empty_session_reports_no_coverage() -> None:
    """Nothing was tracked, so nothing is masked: 0%, not the schema's 100%."""
    summary = session_quality({"n_frames": 12, "scores": []})
    assert summary["n_frames"] == 12
    assert summary["n_masked"] == 0
    assert summary["coverage"] == 0.0


def test_the_manifest_warns_when_frames_have_no_mask() -> None:
    quality = session_quality(
        {
            "n_frames": 30,
            "prompt_frames": [0],
            "absent_frames": [1, 2, 3],
            "scores": scores_for(30, absent=[1, 2, 3], low=[], prompted=[0]),
        }
    )
    manifest = build_manifest(
        "overlay_mp4",
        n_frames=30,
        fps=30.0,
        width=320,
        height=240,
        source_width=320,
        source_height=240,
        has_audio=False,
        audio_preserved=False,
        options={},
        quality=quality,
        session_id="abc",
        model="naive",
    )
    notes = " ".join(manifest["notes"])
    assert "27 of 30 frames have a mask" in notes
    assert "review them in the timeline" in notes


def test_the_manifest_talks_about_weak_frames_when_none_are_missing() -> None:
    """"No mask" is a claim to earn: without absent frames the warning must not say it."""
    quality = session_quality(
        {
            "n_frames": 30,
            "prompt_frames": [0],
            "absent_frames": [],
            "scores": scores_for(30, absent=[], low=[5, 6], prompted=[0]),
        }
    )
    manifest = build_manifest(
        "overlay_mp4",
        n_frames=30,
        fps=30.0,
        width=320,
        height=240,
        options={},
        quality=quality,
        session_id="abc",
        model="naive",
    )
    notes = " ".join(manifest["notes"])
    assert "30 of 30 frames have a mask" in notes
    assert "2 frames below the confidence threshold" in notes
    assert "Frames with no mask show" not in notes
