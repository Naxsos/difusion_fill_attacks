"""Step schedules: *when* (at which denoising steps) an intervention fires.

DiffusionGemma denoises each canvas over ``total_steps`` iterations. Internally the
loop counts DOWN (``cur_step`` goes ``N -> 1``); we expose a friendlier **0-indexed
from the start** convention: ``step_idx = 0`` is the first denoising step, and the
processor converts via ``step_idx = total_steps - cur_step``.

A ``StepSchedule`` is just a predicate over ``(step_idx, total_steps)``. New schedules
(a range, every-other step, all steps, ...) can be added by subclassing without
touching the processor or the rest of the library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class StepSchedule(ABC):
    """Decides whether an intervention is active at a given denoising step."""

    @abstractmethod
    def contains(self, step_idx: int, total_steps: int) -> bool:
        """Return True if the intervention should fire at ``step_idx`` (0-indexed)."""
        raise NotImplementedError


@dataclass(frozen=True)
class PerturbAt(StepSchedule):
    """Fire at exactly one step (``mode="perturb"``).

    A single-step nudge: whether it survives to the emitted token is emergent, so this
    is the schedule for studying how an intervention propagates through denoising.
    """

    step: int

    def contains(self, step_idx: int, total_steps: int) -> bool:
        return step_idx == self.step


@dataclass(frozen=True)
class PinFrom(StepSchedule):
    """Fire at ``step`` and every later step through the end (``mode="pin"``).

    Pinning to the end is what makes a hard freeze stick: combined with ``k=1`` /
    ``p=1.0`` (or any ``p > 0.5``) the final step's argmax is the target token, so the
    emitted token is guaranteed.
    """

    step: int

    def contains(self, step_idx: int, total_steps: int) -> bool:
        return step_idx >= self.step


def schedule_from_mode(mode: str, step: int) -> StepSchedule:
    """Build a schedule from the ``mode`` string used by the high-level ``steer`` API."""
    if mode == "perturb":
        return PerturbAt(step)
    if mode == "pin":
        return PinFrom(step)
    raise ValueError(f"Unknown mode {mode!r}; expected 'perturb' or 'pin'.")
