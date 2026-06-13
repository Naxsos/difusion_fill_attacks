"""The two ``LogitsProcessor`` engines: intervention injection and step recording.

Both are plugged into ``model.generate(logits_processor=...)`` and are called once per
denoising step as ``__call__(input_ids, scores, cur_step)`` with ``scores`` shaped
``(batch, canvas_length, vocab)``. ``LogitsProcessorList`` forwards ``cur_step`` because
the signature takes a third positional argument.

Temperature handling (the crux of exact probability control)
------------------------------------------------------------
DiffusionGemma's built-in ``LinearTemperatureScheduleLogitsProcessor`` runs *after* our
processors and divides every logit by ``T(cur_step) = t_min + (t_max - t_min) *
cur_step / total_steps`` (generation_diffusion_gemma.py:311). So to make the *effective*
(post-temperature) distribution exactly ``D``, ``InterventionLogitsProcessor`` writes
``T * log(D)``: the later ``/ T`` recovers ``log(D)``, whose softmax is ``D``.

``StepRecorder`` runs before that division too, so to report the distribution the
sampler actually sees it re-applies ``/ T`` itself.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor

from .intervention import Intervention


def compute_temperature(
    cur_step: int, t_min: float | None, t_max: float | None, total_steps: int
) -> float:
    """Mirror ``LinearTemperatureScheduleLogitsProcessor``; 1.0 if no schedule is active."""
    if t_min is None or t_max is None:
        return 1.0
    return t_min + (t_max - t_min) * (cur_step / total_steps)


def canvas_index(input_ids: torch.Tensor, prompt_len: int, canvas_length: int) -> int:
    """Which output canvas (block of ``canvas_length``) is currently being denoised."""
    return (input_ids.shape[1] - prompt_len) // canvas_length


class InterventionLogitsProcessor(LogitsProcessor):
    """Injects target distributions at the configured positions/steps/canvases."""

    def __init__(
        self,
        interventions: Sequence[Intervention],
        *,
        prompt_len: int,
        canvas_length: int,
        total_steps: int,
        t_min: float | None,
        t_max: float | None,
    ):
        self.interventions = list(interventions)
        self.prompt_len = prompt_len
        self.canvas_length = canvas_length
        self.total_steps = total_steps
        self.t_min = t_min
        self.t_max = t_max

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, cur_step
    ) -> torch.FloatTensor:
        cur = int(cur_step)
        step_idx = self.total_steps - cur
        c_idx = canvas_index(input_ids, self.prompt_len, self.canvas_length)
        temp = compute_temperature(cur, self.t_min, self.t_max, self.total_steps)

        for iv in self.interventions:
            if iv.position // self.canvas_length != c_idx:
                continue
            if not iv.schedule.contains(step_idx, self.total_steps):
                continue
            local = iv.position % self.canvas_length
            row = scores[:, local, :]  # (batch, vocab), raw logits at this position
            dist = iv.policy.target_distribution(row, iv.token_id, iv.p, iv.k)
            # T * log(D): the downstream /T recovers log(D) -> softmax == D. log(0) -> -inf.
            log_dist = torch.log(dist)
            scores[:, local, :] = (temp * log_dist).to(scores.dtype)

        return scores


class EosSuppressionLogitsProcessor(LogitsProcessor):
    """Bans EOS tokens at every output position up to ``until_position`` (inclusive).

    DiffusionGemma otherwise emits EOS at its natural ending and pads everything after,
    so interventions at far positions get appended past a "finished" sequence (or padded
    away). Forcing EOS to ``-inf`` for all positions through the last intervention makes
    the model generate real content right into the steered region instead of stopping
    early. EOS is left free *after* ``until_position`` so the model can still end normally.
    """

    def __init__(
        self,
        eos_token_ids: Sequence[int],
        until_position: int,
        *,
        prompt_len: int,
        canvas_length: int,
    ):
        self.eos_token_ids = torch.as_tensor(list(eos_token_ids), dtype=torch.long)
        self.until_position = until_position
        self.prompt_len = prompt_len
        self.canvas_length = canvas_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, cur_step
    ) -> torch.FloatTensor:
        c_idx = canvas_index(input_ids, self.prompt_len, self.canvas_length)
        base = c_idx * self.canvas_length  # output position of this canvas's local index 0
        last_local = self.until_position - base
        if last_local < 0:
            return scores  # this whole canvas is past the cutoff; leave EOS free
        last_local = min(last_local, self.canvas_length - 1)
        eos = self.eos_token_ids.to(scores.device)
        # Fill the EOS vocab columns with -inf for local positions 0..last_local.
        scores[:, : last_local + 1, :].index_fill_(-1, eos, float("-inf"))
        return scores


@dataclass
class RecorderConfig:
    """What ``StepRecorder`` captures each step (kept small to avoid OOM by default)."""

    positions: Sequence[int] | None = None  # canvas positions for top-k; None = all
    top_k: int = 50
    record_argmax: bool = True
    record_entropy: bool = True
    record_full_logits: bool = False  # whole-vocab dump; multi-GB, opt-in only
    post_temperature: bool = True  # report the distribution the sampler actually sees
    batch_index: int = 0  # v1 records a single sequence


class StepRecorder(LogitsProcessor):
    """Pass-through processor that records per-step probabilities/argmax/entropy.

    Records to CPU so GPU memory doesn't grow across steps. Access the captured data via
    ``.records`` (a list of dicts, one per denoising step, in call order).
    """

    def __init__(
        self,
        config: RecorderConfig,
        *,
        prompt_len: int,
        canvas_length: int,
        total_steps: int,
        t_min: float | None,
        t_max: float | None,
    ):
        self.config = config
        self.prompt_len = prompt_len
        self.canvas_length = canvas_length
        self.total_steps = total_steps
        self.t_min = t_min
        self.t_max = t_max
        self.records: list[dict] = []
        if config.record_full_logits:
            warnings.warn(
                "RecorderConfig.record_full_logits dumps the whole vocab for every "
                "position and step (multiple GB on CPU). Use only for short runs.",
                stacklevel=2,
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, cur_step
    ) -> torch.FloatTensor:
        cur = int(cur_step)
        cfg = self.config
        temp = compute_temperature(cur, self.t_min, self.t_max, self.total_steps)

        # (canvas_length, vocab) float32 logits for the recorded sequence, post-temp if asked.
        logits = scores[cfg.batch_index].float()
        if cfg.post_temperature:
            logits = logits / temp

        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)  # (canvas_length, 1)
        record: dict = {
            "cur_step": cur,
            "step_idx": self.total_steps - cur,
            "canvas_idx": canvas_index(input_ids, self.prompt_len, self.canvas_length),
        }

        if cfg.record_argmax:
            record["argmax"] = logits.argmax(dim=-1).cpu()  # (canvas_length,)
        if cfg.record_entropy:
            log_p = logits - log_z
            record["entropy"] = -(log_p.exp() * log_p).sum(dim=-1).cpu()  # (canvas_length,)

        # Top-k token ids + probabilities for the requested positions (default: all).
        pos = (
            torch.arange(self.canvas_length)
            if cfg.positions is None
            else torch.as_tensor(list(cfg.positions))
        )
        sel = logits[pos]  # (n_pos, vocab)
        top_vals, top_idx = torch.topk(sel, min(cfg.top_k, sel.shape[-1]), dim=-1)
        record["positions"] = pos.cpu()
        record["topk_ids"] = top_idx.cpu()
        record["topk_probs"] = (top_vals - log_z[pos]).exp().cpu()  # softmax over vocab

        if cfg.record_full_logits:
            record["logits"] = logits.half().cpu()

        self.records.append(record)
        return scores
