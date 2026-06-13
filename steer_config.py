"""Typed config for a single steering run -- a friendlier stand-in for argparse args.

``SteerConfig`` carries exactly the knobs ``example_steer``'s CLI exposes, and its field
names match the CLI's argparse *dests* (``start_pos``, ``trace_topk``, ...). That means
``example_steer.run_experiment`` can take either a ``SteerConfig`` or an argparse
``Namespace`` interchangeably -- both are plain attribute bags with the same names -- so
existing CLI usage keeps working while batch/programmatic callers get a typed object
instead of juggling argument strings.

``example_steer`` sources its parser defaults from ``SteerConfig()`` too, so the dataclass
is the single source of truth for defaults.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

DEFAULT_PROMPT = "Count up from one: 1, 2, 3,"


@dataclass
class SteerConfig:
    """One steering experiment's settings. See ``example_steer`` for the full semantics.

    ``target``/``start_pos``/``mode``/``step`` are per-target parallel lists (a length-1
    list broadcasts to every target). ``prob``/``k`` broadcast across all pinned tokens.
    """

    # What to inject and where.
    prompt: str = DEFAULT_PROMPT
    target: list[str] = field(default_factory=lambda: [" 9, 8, 7"])
    start_pos: list[int] = field(default_factory=lambda: [0])
    prob: float | None = None  # per-token probability; None = hard pin
    k: int = 1  # top-k width (1 = hard freeze)

    # When to inject (per target; see PinFrom/PerturbAt schedules).
    mode: list[str] = field(default_factory=lambda: ["pin"])
    step: list[int] = field(default_factory=lambda: [0])

    # Per-step recording / output.
    trace: bool = False
    trace_topk: int = 5
    trace_positions: list[int] | None = None
    trace_file: str | None = None

    # Where the server lives.
    host: str = "localhost"
    port: int = 8000


def to_cli(cfg: SteerConfig, script: str = "example_steer.py") -> str:
    """Render a ``SteerConfig`` as the equivalent runnable ``example_steer`` command.

    Used to record, alongside each experiment's output, the exact command that produced
    it -- so a saved run is unambiguous and reproducible. Non-default knobs are emitted;
    list-valued flags expand to space-separated, shell-quoted values.
    """
    parts: list[str] = ["python", script]

    def add(flag: str, values) -> None:
        parts.append(flag)
        parts.extend(shlex.quote(str(v)) for v in values)

    add("--prompt", [cfg.prompt])
    add("--target", cfg.target)
    add("--start-pos", cfg.start_pos)
    if cfg.prob is not None:
        add("--prob", [cfg.prob])
    add("--k", [cfg.k])
    add("--mode", cfg.mode)
    add("--step", cfg.step)
    if cfg.trace_positions is not None:
        add("--trace-positions", cfg.trace_positions)
    if cfg.trace_file is not None:
        add("--trace-file", [cfg.trace_file])
    elif cfg.trace:
        parts.append("--trace")
    if cfg.host != "localhost":
        add("--host", [cfg.host])
    if cfg.port != 8000:
        add("--port", [cfg.port])
    return " ".join(parts)
