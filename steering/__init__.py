"""Steering / intervention library for DiffusionGemma.

Force a chosen token at a chosen output position, with a chosen probability, at a chosen
denoising step -- and read per-step logits/probabilities. See the package modules for
the mechanics; the typical entry point is ``steer``.

Quickstart
----------
    from load_model import load_model
    from steering import steer

    model, processor = load_model()

    # Freeze the first generated token to " 5" through the end of denoising:
    result = steer(model, processor, "Count up from 1.",
                   tokens=[" 5"], positions=[0], ks=1, modes="pin")
    print(result.summary())

Extension points (subclass and pass in, no core changes needed):
  * ``DistributionPolicy`` -- what target distribution to inject (default: top-k ∝ model).
  * ``StepSchedule``       -- when an intervention fires (``PerturbAt`` / ``PinFrom``).
  * ``RecorderConfig``     -- what per-step data to capture.
"""

from __future__ import annotations

from .api import InterventionOutcome, SteerResult, steer
from .intervention import (
    Intervention,
    build_interventions,
    make_intervention,
    resolve_token_id,
)
from .policies import DistributionPolicy, TopKProportionalPolicy
from .processor import (
    EosSuppressionLogitsProcessor,
    InterventionLogitsProcessor,
    RecorderConfig,
    StepRecorder,
    compute_temperature,
)
from .schedule import PerturbAt, PinFrom, StepSchedule, schedule_from_mode

__all__ = [
    # high-level
    "steer",
    "SteerResult",
    "InterventionOutcome",
    # interventions
    "Intervention",
    "make_intervention",
    "build_interventions",
    "resolve_token_id",
    # policies
    "DistributionPolicy",
    "TopKProportionalPolicy",
    # schedules
    "StepSchedule",
    "PerturbAt",
    "PinFrom",
    "schedule_from_mode",
    # processors / recording
    "InterventionLogitsProcessor",
    "EosSuppressionLogitsProcessor",
    "StepRecorder",
    "RecorderConfig",
    "compute_temperature",
]
