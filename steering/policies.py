"""Distribution policies: *what* target distribution to inject at a position.

A ``DistributionPolicy`` turns the model's raw logits at one canvas position into a
target probability distribution that the intervention will force. The policy is
stateless -- the per-intervention knobs (``p``, ``k``) are passed in -- so a single
instance can be shared across interventions.

The default ``TopKProportionalPolicy`` implements the agreed semantics:
  * ``k == 1``           -> hard point mass on the target (``p`` forced to 1.0).
  * ``k >= 2``           -> target gets ``p``; the residual ``1 - p`` is spread over the
                            top ``k - 1`` runner-up tokens (excluding the target),
                            **proportional to the model's own probabilities**.

New strategies (uniform residual, entropy-targeted, additive logit bias, suppression)
subclass ``DistributionPolicy`` without touching the processor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DistributionPolicy(ABC):
    """Maps a raw logits row + target/p/k to a normalized target distribution."""

    @abstractmethod
    def target_distribution(
        self, raw_logits_row: torch.Tensor, target_id: int, p: float, k: int
    ) -> torch.Tensor:
        """Return probabilities of shape ``(batch, vocab)`` that sum to 1 per row.

        Args:
            raw_logits_row: ``(batch, vocab)`` raw model logits at the target position.
            target_id:      vocab id to steer toward.
            p:              probability mass on ``target_id`` (ignored when ``k == 1``).
            k:              number of tokens carrying mass (target + ``k - 1`` runner-ups).
        """
        raise NotImplementedError


class TopKProportionalPolicy(DistributionPolicy):
    """Target gets ``p``; residual spread over top ``k-1`` runner-ups, ∝ model probs."""

    def target_distribution(
        self, raw_logits_row: torch.Tensor, target_id: int, p: float, k: int
    ) -> torch.Tensor:
        batch, vocab = raw_logits_row.shape
        dist = torch.zeros((batch, vocab), dtype=torch.float32, device=raw_logits_row.device)

        # k == 1 (or no residual mass): a hard point mass on the target.
        if k <= 1 or p >= 1.0:
            dist[:, target_id] = 1.0
            return dist

        # k >= 2: pick the top (k-1) runner-ups by raw logits, excluding the target.
        logits = raw_logits_row.float().clone()
        logits[:, target_id] = float("-inf")
        num_runners = min(k - 1, vocab - 1)
        top_vals, top_idx = torch.topk(logits, num_runners, dim=-1)  # (batch, k-1)

        # Split the residual ∝ the model's own probabilities among those runner-ups.
        runner_weights = torch.softmax(top_vals, dim=-1)  # (batch, k-1), sums to 1
        dist[:, target_id] = p
        dist.scatter_(1, top_idx, runner_weights * (1.0 - p))
        return dist
