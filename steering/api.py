"""High-level ``steer`` entry point and its result types.

``steer`` is the one function most experiments call. It accepts the requested
parallel-array form -- ``tokens``, ``positions``, ``probabilities``, ``steps`` -- (or a
prebuilt list of ``Intervention`` objects), runs a single steered generation, and
returns a ``SteerResult`` reporting the decoded text, what token actually landed at each
intervened position (and whether the steer ``held``), and an optional per-step trace.

This runs **in-process**: it needs the model loaded in the current process (see
``load_model.load_model``), not the HTTP server.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import LogitsProcessorList

from .intervention import Intervention, build_interventions
from .processor import (
    EosSuppressionLogitsProcessor,
    InterventionLogitsProcessor,
    RecorderConfig,
    StepRecorder,
)

# Schedule defaults that match DiffusionGemma's shipped generation_config.
_DEFAULT_TOTAL_STEPS = 48
_DEFAULT_T_MIN = 0.4
_DEFAULT_T_MAX = 0.8


@dataclass
class InterventionOutcome:
    """What actually happened at one intervened position after generation."""

    position: int
    requested_id: int
    label: int | str | None
    actual_id: int | None
    actual_token: str | None
    held: bool
    p: float
    k: int


@dataclass
class SteerResult:
    """Result of a steered generation."""

    text: str
    sequences: torch.LongTensor  # (batch, total_len), prompt included
    new_token_ids: torch.LongTensor  # generated tokens only (prompt stripped)
    interventions: list[InterventionOutcome]
    trace: list[dict] | None  # per-step recorder records, or None

    @property
    def all_held(self) -> bool:
        return all(o.held for o in self.interventions)

    def summary(self) -> str:
        lines = [f"text: {self.text!r}"]
        for o in self.interventions:
            mark = "ok" if o.held else "MISS"
            lines.append(
                f"  [{mark}] pos {o.position}: wanted {o.label!r} (id {o.requested_id}, "
                f"p={o.p}, k={o.k}) -> got {o.actual_token!r} (id {o.actual_id})"
            )
        return "\n".join(lines)


def _resolve_schedule_params(model, max_denoising_steps: int | None) -> tuple[int, float, float]:
    """Pull total_steps / t_min / t_max from the model's generation config, with defaults."""
    gen_cfg = getattr(model, "generation_config", None)
    total_steps = max_denoising_steps or getattr(gen_cfg, "max_denoising_steps", None) or _DEFAULT_TOTAL_STEPS
    t_min = getattr(gen_cfg, "t_min", None)
    t_max = getattr(gen_cfg, "t_max", None)
    t_min = _DEFAULT_T_MIN if t_min is None else t_min
    t_max = _DEFAULT_T_MAX if t_max is None else t_max
    return int(total_steps), float(t_min), float(t_max)


def steer(
    model,
    processor,
    prompt: str,
    tokens: Sequence[int | str] | None = None,
    positions: Sequence[int] | None = None,
    probabilities: Sequence[float | None] | float | None = None,
    steps: Sequence[int] | int = 0,
    ks: Sequence[int] | int = 2,
    modes: Sequence[str] | str = "pin",
    *,
    interventions: Sequence[Intervention] | None = None,
    max_new_tokens: int = 256,
    max_denoising_steps: int | None = None,
    disable_adaptive_stopping: bool = True,
    keep_after_eos: bool = False,
    suppress_eos_until: int | bool | None = None,
    record: RecorderConfig | bool | None = None,
    seed: int | None = None,
) -> SteerResult:
    """Run one steered generation.

    Args:
        model, processor: as returned by ``load_model.load_model``.
        prompt: user turn; wrapped with the chat template.
        tokens, positions, probabilities, steps, ks, modes: parallel-array interventions.
            ``positions`` are 0-indexed in the generated output (0 = first new token).
            Scalars broadcast across all positions. Ignored if ``interventions`` is given.
        interventions: prebuilt ``Intervention`` list, as an alternative to the arrays.
        max_new_tokens: rounded up to whole 256-token canvases by the model.
        max_denoising_steps: fixed step count (defaults to the model's config, 48).
        disable_adaptive_stopping: keep step indices stable/reproducible (recommended).
        keep_after_eos: if False (default), the model may emit EOS and the generation loop
            overwrites every position after it with ``<pad>`` -- so interventions at
            positions past the model's natural ending silently vanish (``held`` is False).
            Set True to drop ``eos_token_id``, which disables both the early stop and the
            post-EOS padding, so far positions are actually realized. Cost: it always
            generates the full ``max_new_tokens`` and emits filler after the natural end.
        suppress_eos_until: ban EOS tokens at every output position up to (and including)
            this one, so the model can't end before the steered region -- it generates
            real content right into it, then may end normally after. ``True`` auto-uses the
            last intervention position (recommended for far-position steering); an int sets
            an explicit cutoff; ``None``/``False`` disables. Usually preferable to
            ``keep_after_eos`` because EOS-terminated padding then cleanly trims only the
            tail *after* your interventions.
        record: ``True`` for a default recorder, a ``RecorderConfig`` for a custom one,
            or ``None``/``False`` for no trace.
        seed: seeds the RNG so the multinomial sampling in denoising is reproducible.

    Returns:
        SteerResult.
    """
    if interventions is None:
        if tokens is None or positions is None:
            raise ValueError("Provide either `interventions=` or both `tokens=` and `positions=`.")
        interventions = build_interventions(
            tokens, positions, processor=processor,
            probabilities=probabilities, steps=steps, ks=ks, modes=modes,
        )
    interventions = list(interventions)

    message = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        message, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    canvas_length = model.config.canvas_length
    total_steps, t_min, t_max = _resolve_schedule_params(model, max_denoising_steps)

    shared = dict(
        prompt_len=prompt_len, canvas_length=canvas_length,
        total_steps=total_steps, t_min=t_min, t_max=t_max,
    )
    processors: list[Any] = []
    if suppress_eos_until is not None and suppress_eos_until is not False and interventions:
        cutoff = (
            max(iv.position for iv in interventions)
            if suppress_eos_until is True
            else int(suppress_eos_until)
        )
        eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos_ids is not None:
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            processors.append(
                EosSuppressionLogitsProcessor(
                    eos_ids, cutoff, prompt_len=prompt_len, canvas_length=canvas_length
                )
            )
    processors.append(InterventionLogitsProcessor(interventions, **shared))
    recorder: StepRecorder | None = None
    if record:
        cfg = record if isinstance(record, RecorderConfig) else RecorderConfig()
        recorder = StepRecorder(cfg, **shared)
        processors.append(recorder)

    if seed is not None:
        torch.manual_seed(seed)

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        max_denoising_steps=total_steps,
        t_min=t_min, t_max=t_max,
        logits_processor=LogitsProcessorList(processors),
    )
    if disable_adaptive_stopping:
        gen_kwargs.update(stability_threshold=None, confidence_threshold=None)
    if keep_after_eos:
        # Dropping eos_token_id removes the EosTokenCriteria (no early stop) and the
        # post-EOS <pad> overwrite in _finalize_canvas, so interventions past the model's
        # natural ending actually land instead of being padded away.
        gen_kwargs["eos_token_id"] = None

    output = model.generate(**inputs, **gen_kwargs)

    new_ids = output.sequences[0][prompt_len:]
    text = processor.decode(new_ids, skip_special_tokens=True).strip()

    outcomes: list[InterventionOutcome] = []
    for iv in interventions:
        actual_id = int(new_ids[iv.position]) if iv.position < new_ids.shape[0] else None
        actual_token = processor.decode([actual_id]) if actual_id is not None else None
        outcomes.append(
            InterventionOutcome(
                position=iv.position, requested_id=iv.token_id, label=iv.label,
                actual_id=actual_id, actual_token=actual_token,
                held=(actual_id == iv.token_id), p=iv.p, k=iv.k,
            )
        )

    return SteerResult(
        text=text,
        sequences=output.sequences,
        new_token_ids=new_ids,
        interventions=outcomes,
        trace=recorder.records if recorder is not None else None,
    )
