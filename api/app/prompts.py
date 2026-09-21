"""Translate between API prompt schemas and the tracker contract's ``PromptSet``.

The HTTP layer speaks ``PromptIn`` (pydantic, JSON-friendly); the model contract
speaks ``PromptSet`` (frozen dataclasses, framework-free). Keeping the conversion
in one place means the model boundary never imports the web layer.
"""
from __future__ import annotations

from .models.base import BoxPrompt, PointPrompt, PromptSet
from .schemas import PromptIn


class PromptError(ValueError):
    """A prompt that cannot be turned into a valid PromptSet."""


def to_prompt_set(prompt: PromptIn, frame_index: int | None = None) -> PromptSet:
    points = tuple(
        PointPrompt(x=float(p.x), y=float(p.y), positive=bool(p.positive))
        for p in prompt.points
    )
    box = (
        BoxPrompt(
            x0=float(prompt.box.x0), y0=float(prompt.box.y0),
            x1=float(prompt.box.x1), y1=float(prompt.box.y1),
        )
        if prompt.box is not None
        else None
    )
    if not points and box is None:
        raise PromptError("a prompt needs at least one point or a box")

    index = prompt.frame_index if frame_index is None else frame_index
    if index < 0:
        raise PromptError("frame_index must be >= 0")
    return PromptSet(frame_index=int(index), points=points, box=box)


def prompt_set_to_dict(prompt: PromptSet) -> dict:
    """Serialise a PromptSet back to JSON (used in session metadata)."""
    return {
        "frame_index": prompt.frame_index,
        "points": [
            {"x": p.x, "y": p.y, "positive": p.positive} for p in prompt.points
        ],
        "box": (
            {
                "x0": prompt.box.x0, "y0": prompt.box.y0,
                "x1": prompt.box.x1, "y1": prompt.box.y1,
            }
            if prompt.box is not None
            else None
        ),
    }
