"""The ``Intervention`` value object plus helpers to build interventions ergonomically.

An ``Intervention`` ties together *what* token to inject (``token_id``), *where*
(``position`` in the generated output, 0 = first generated token), *how strongly*
(``p``, ``k`` + a ``DistributionPolicy``) and *when* (a ``StepSchedule``).

Tokens may be given as int ids or single-token strings; strings are resolved against
the processor's tokenizer and rejected if they don't encode to exactly one token.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .policies import DistributionPolicy, TopKProportionalPolicy
from .schedule import StepSchedule, schedule_from_mode

# Shared default policy instance (stateless, so one is enough).
_DEFAULT_POLICY = TopKProportionalPolicy()


def resolve_token_id(token: int | str, processor: Any) -> int:
    """Resolve an int id or a single-token string to a vocab id.

    Raises ValueError if a string does not encode to exactly one token, so a caller
    never silently steers toward a different token than they intended.
    """
    if isinstance(token, bool):  # bool is an int subclass; almost certainly a mistake
        raise TypeError(f"token must be an int id or str, got bool {token!r}")
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        tokenizer = getattr(processor, "tokenizer", processor)
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"token {token!r} encodes to {len(ids)} tokens ({ids}); pass a "
                f"single-token string (mind leading spaces, e.g. ' 5' vs '5') or an int id."
            )
        return ids[0]
    raise TypeError(f"token must be an int id or str, got {type(token).__name__}")


@dataclass
class Intervention:
    """A single steering intervention at one output position."""

    token_id: int
    position: int
    p: float
    k: int
    schedule: StepSchedule
    policy: DistributionPolicy = field(default_factory=lambda: _DEFAULT_POLICY)
    label: int | str | None = None  # original token spec, kept for reporting

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be an int >= 1, got {self.k!r}")
        if not (0.0 < self.p <= 1.0):
            raise ValueError(f"p must be in (0, 1], got {self.p!r}")
        if self.position < 0:
            raise ValueError(f"position must be >= 0, got {self.position!r}")


def make_intervention(
    token: int | str,
    position: int,
    *,
    processor: Any,
    p: float | None = None,
    k: int = 2,
    mode: str = "pin",
    step: int = 0,
    policy: DistributionPolicy | None = None,
) -> Intervention:
    """Build one ``Intervention``, resolving the token and the ``mode``/``step`` schedule.

    ``p`` defaults to 1.0 (a hard freeze). For ``k == 1`` the policy forces ``p = 1.0``
    regardless, so passing ``p`` there is unnecessary.
    """
    resolved_p = 1.0 if p is None else p
    return Intervention(
        token_id=resolve_token_id(token, processor),
        position=position,
        p=resolved_p,
        k=k,
        schedule=schedule_from_mode(mode, step),
        policy=policy or _DEFAULT_POLICY,
        label=token,
    )


def _broadcast(value: Any, n: int, name: str) -> list:
    """Turn a scalar into a length-n list, or validate an existing sequence's length."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name} has length {len(value)}, expected {n}")
    return list(value)


def build_interventions(
    tokens: Sequence[int | str],
    positions: Sequence[int],
    *,
    processor: Any,
    probabilities: Sequence[float | None] | float | None = None,
    steps: Sequence[int] | int = 0,
    ks: Sequence[int] | int = 2,
    modes: Sequence[str] | str = "pin",
    policy: DistributionPolicy | None = None,
) -> list[Intervention]:
    """Build a list of interventions from parallel arrays (the ``steer`` API form).

    ``tokens`` and ``positions`` set the count ``n``; every other argument is either a
    scalar (broadcast to all ``n``) or a length-``n`` sequence.
    """
    if len(tokens) != len(positions):
        raise ValueError(f"tokens ({len(tokens)}) and positions ({len(positions)}) must match")
    n = len(tokens)
    probs = _broadcast(probabilities, n, "probabilities")
    step_list = _broadcast(steps, n, "steps")
    k_list = _broadcast(ks, n, "ks")
    mode_list = _broadcast(modes, n, "modes")

    return [
        make_intervention(
            token, pos, processor=processor, p=prob, k=k, mode=mode, step=step, policy=policy
        )
        for token, pos, prob, step, k, mode in zip(
            tokens, positions, probs, step_list, k_list, mode_list
        )
    ]
