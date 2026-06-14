"""Steering example: target a whole STRING, let the tokenizer place the tokens.

Instead of hand-picking single-token strings (and guessing whether "5" vs " 5" is one
token), you give a *target string*. We tokenize it locally, then pin each resulting
token to consecutive output positions starting at `start_pos`:

    positions = [start_pos, start_pos + 1, ..., start_pos + len(token_ids) - 1]

The token ids are passed straight to the server's /steer endpoint (it accepts int ids,
so the "does this string encode to exactly one token?" check never trips). The heavy
52 GB model stays on the server -- here we only load the lightweight *tokenizer*, so this
still needs no GPU (just `transformers`, which the .venv already has).

    python example_steer.py
    python example_steer.py --host a100-box --target " 4, 5, 6" --start-pos 0

You can also inject several disjoint spans at once by passing parallel lists -- one
start position per target string:

    python example_steer.py --target " 4, 5, 6" " END" --start-pos 0 10

Steering one position reshapes the WHOLE completion, because a diffusion model fills
every position jointly -- so a steer is really a "fill attack". Plant a verdict at the
first output position and the model writes reasoning that rationalizes it (compare the
printed baseline, which often picks the opposite verdict):

    python example_steer.py \
        --prompt "Is a hot dog a sandwich? Give a one-word verdict (Yes or No), then explain." \
        --target "Yes" --start-pos 0

Now add per-target *timing* to hijack the answer mid-thought: pin one verdict from step 0,
then override it with the opposite verdict at step 25. Trace position 0 to see exactly
when the canvas commits -- the final verdict flips late, but the explanation (already
denoised to support the first verdict) is often left arguing the other way, a striking
self-contradiction that exposes how/when the model "decided":

    python example_steer.py \
        --prompt "Is a hot dog a sandwich? Give a one-word verdict (Yes or No), then explain." \
        --target "Yes" "No" --start-pos 0 0 --step 0 25 \
        --trace-positions 0 --trace-file verdict.json

The terminal shows a compact trajectory like  pos 0: 'Yes'[0-24]  ->  'No'[25-47];
verdict.json holds the full top-k-per-step trace for tracing the reasoning back.

Note on leading spaces: a string tokenizes the same way it would mid-sentence, so a
number that appears after a space in the output (e.g. "... 3, 4") usually wants a leading
space in the target (" 4, 5, 6"), not "4, 5, 6". Inspect the printed id->token mapping to
check you're pinning what you think you are.
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoProcessor

from client import steer
from load_model import DEFAULT_MODEL_ID
from steer_config import DEFAULT_PROMPT, SteerConfig

PROMPT = DEFAULT_PROMPT  # kept as a module-level alias for backward compatibility


def load_tokenizer():
    """Load just the tokenizer (no model weights, no GPU) for client-side tokenizing."""
    return AutoProcessor.from_pretrained(DEFAULT_MODEL_ID).tokenizer


def _per_target(value, n: int, name: str) -> list:
    """Normalize a per-target arg to a length-`n` list: a scalar (or length-1 list)
    broadcasts to every target; a longer list must already match the target count."""
    if not isinstance(value, (list, tuple)):
        return [value] * n
    if len(value) == 1:
        return list(value) * n
    if len(value) != n:
        raise ValueError(f"{name} has length {len(value)}, expected 1 or {n} (one per target)")
    return list(value)


def steer_strings(
    prompt: str,
    targets,
    start_positions,
    tokenizer,
    *,
    probabilities=None,
    ks=1,
    modes: str | list[str] = "pin",
    steps: int | list[int] = 0,
    suppress_eos_until=True,
    seed: int = 0,
    trace: bool = False,
    trace_topk: int = 5,
    trace_positions=None,
    **where,
) -> dict:
    """Pin several `targets`, each at its own `start_pos`, in a single steer call.

    `targets` and `start_positions` are parallel lists: each target string is tokenized
    locally and pinned to consecutive positions from its start_pos. This lets you inject
    multiple disjoint spans (e.g. one at position 0, another later in the output) in one
    generation. A single string / int is accepted too and wrapped automatically.

    `modes`/`steps` control *when* in the denoising process each target is injected (the
    "insert at time" axis) and are **per-target** (one value per target string, applied to
    all of that target's tokens; a scalar broadcasts to every target). `mode="pin",
    step=S` forces the target from denoising step S through the end (so it sticks), while
    `mode="perturb", step=S` nudges it at *only* step S and then releases.

    Because targets are applied in order and a later one overwrites an earlier one at a
    shared position, you can steer at staggered times: pin target A at one position from
    step 10, then pin a *different* target B at the same position from step 20 -- the model
    denoises ~10 steps under A before B takes over (B "undoes" A).

    `probabilities`/`ks` broadcast across *all* pinned tokens (or pass a list matching the
    total token count). `suppress_eos_until=True` (default here) bans EOS up
    to the last pinned position, so the model generates real content right into the
    steered region instead of ending early and appending the target after a "finished"
    sequence. Returns the server's /steer payload, with the resolved `token_ids` and
    `positions` attached for inspection.
    """
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(start_positions, int):
        start_positions = [start_positions]
    if len(targets) != len(start_positions):
        raise ValueError(
            f"need one start position per target: got {len(targets)} targets "
            f"but {len(start_positions)} start positions"
        )
    # modes/steps are per-target: a scalar (or length-1 list) broadcasts to every target.
    modes = _per_target(modes, len(targets), "modes")
    steps = _per_target(steps, len(targets), "steps")

    token_ids: list[int] = []
    positions: list[int] = []
    token_modes: list[str] = []
    token_steps: list[int] = []
    for target, start_pos, mode, step in zip(targets, start_positions, modes, steps):
        ids = tokenizer.encode(target, add_special_tokens=False)
        pos = list(range(start_pos, start_pos + len(ids)))
        print(f"  target {target!r} -> ids {ids} -> positions {pos} (mode={mode}, step={step})")
        print("    " + "  ".join(f"{p}:{tokenizer.decode([i])!r}" for p, i in zip(pos, ids)))
        token_ids.extend(ids)
        positions.extend(pos)
        # Each target's mode/step applies to all of its tokens; expand to per-token lists
        # matching token_ids (build_interventions accepts per-element steps/modes).
        token_modes.extend([mode] * len(ids))
        token_steps.extend([step] * len(ids))

    # When tracing, ask the server's per-step recorder to capture top-k tokens+probs.
    # Default to the steered positions (the interesting ones); recording every canvas
    # position each step is much heavier and rarely what you want to read back.
    record = None
    if trace:
        record = {"positions": list(trace_positions) if trace_positions else sorted(set(positions)),
                  "top_k": trace_topk,
                  # Skip the per-step full-canvas argmax/entropy dumps; we only read the
                  # top-k tokens+probs at the traced positions.
                  "record_argmax": False, "record_entropy": False}

    result = steer(
        prompt,
        tokens=token_ids,
        positions=positions,
        probabilities=probabilities,
        ks=ks,
        modes=token_modes,
        steps=token_steps,
        suppress_eos_until=suppress_eos_until,
        seed=seed,
        record=record,
        **where,
    )
    result["token_ids"] = token_ids
    result["positions"] = positions
    return result


def decode_trace(trace, tokenizer) -> list[dict]:
    """Turn the server's raw per-step trace into a readable, JSON-safe structure.

    Each denoising step becomes a record holding, for every recorded canvas position:
    - ``positions``: post-intervention top-k (what the sampler actually saw)
    - ``pre_positions``: pre-intervention top-k (the model's natural distribution before
      any steering was applied), present only when the recorder was wired to an
      InterventionLogitsProcessor
    - ``steered_positions``: list of output positions actively steered at this step

    Steps are ordered as denoising actually ran (step_idx ascending within each canvas).
    """
    def _decode_topk(positions, topk_ids, topk_probs):
        per_pos = {}
        for i, pos in enumerate(positions):
            per_pos[int(pos)] = [
                {"id": int(tid), "token": tokenizer.decode([int(tid)]), "prob": round(float(p), 5)}
                for tid, p in zip(topk_ids[i], topk_probs[i])
            ]
        return per_pos

    decoded = []
    for rec in sorted(trace, key=lambda r: (r["canvas_idx"], r["step_idx"])):
        positions = rec.get("positions", [])
        per_pos = _decode_topk(positions, rec.get("topk_ids", []), rec.get("topk_probs", []))

        entry: dict = {
            "step_idx": rec["step_idx"],
            "cur_step": rec["cur_step"],
            "canvas_idx": rec["canvas_idx"],
            "positions": per_pos,
            "steered_positions": rec.get("steered_positions", []),
        }

        if "pre_topk_ids" in rec:
            entry["pre_positions"] = _decode_topk(
                positions, rec["pre_topk_ids"], rec["pre_topk_probs"]
            )

        decoded.append(entry)
    return decoded


def print_trace_summary(decoded: list[dict]) -> None:
    """Print a compact per-position trajectory: the top-1 token over denoising steps,
    with consecutive identical tokens collapsed into step ranges.

    This keeps the terminal readable (a line or two per position) while still showing the
    interesting moments -- e.g. a staggered steer reads as ``'7'[10-19] -> '3'[20-47]``,
    making the hand-off from one intervention to a later one obvious. The full per-step
    top-k detail is what gets written to the trace file.
    """
    # position -> ordered list of (step_idx, top_token, top_prob)
    by_pos: dict[int, list[tuple[int, str, float]]] = {}
    for rec in decoded:
        for pos, cands in rec["positions"].items():
            if cands:
                by_pos.setdefault(pos, []).append((rec["step_idx"], cands[0]["token"], cands[0]["prob"]))

    print("\nTop-1 token trajectory by denoising step (consecutive repeats collapsed):")
    for pos in sorted(by_pos):
        runs: list[list] = []  # each: [token, start_step, end_step, last_prob]
        for step, tok, prob in by_pos[pos]:
            if runs and runs[-1][0] == tok:
                runs[-1][2], runs[-1][3] = step, prob
            else:
                runs.append([tok, step, step, prob])
        desc = "  ->  ".join(
            f"{tok!r}[{s}]" if s == e else f"{tok!r}[{s}-{e}]"
            for tok, s, e, _ in runs
        )
        print(f"  pos {pos:>3}: {desc}")


def write_trace_file(path: str, payload: dict) -> None:
    """Write the run + decoded per-step trace to `path` as JSON for later analysis."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote trace to {path}")


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser for one steering run (shared by `main` and batch runners).

    Defaults come from `SteerConfig()`, so the dataclass and the CLI stay in lockstep.
    """
    d = SteerConfig()  # single source of truth for the defaults below
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=d.host)
    parser.add_argument("--port", type=int, default=d.port)
    parser.add_argument("--prompt", default=d.prompt)
    parser.add_argument(
        "--target", nargs="+", default=d.target,
        help="string(s) to force into the output; pass several to inject disjoint spans",
    )
    parser.add_argument(
        "--start-pos", nargs="+", type=int, default=d.start_pos,
        help="output position of each target's first token (one per --target)",
    )
    parser.add_argument("--prob", type=float, default=d.prob, help="per-token probability (default: hard pin)")
    parser.add_argument("--k", type=int, default=d.k, help="top-k width (1 = hard freeze)")
    parser.add_argument(
        "--mode", nargs="+", choices=("pin", "perturb"), default=d.mode,
        help="per-target (one per --target; a single value broadcasts). "
             "'pin' = inject from --step through the end (sticks); "
             "'perturb' = inject at ONLY --step, then release (survival is emergent)",
    )
    parser.add_argument(
        "--step", nargs="+", type=int, default=d.step,
        help="per-target denoising step to inject at (0-indexed from start; total ~48; "
             "one per --target, a single value broadcasts). Steer at staggered times by "
             "pinning different targets at the same position from different steps -- the "
             "later target overrides the earlier one once both are active",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="record + print the top tokens/probabilities at every denoising step, "
             "so you can track back through how each position firmed up",
    )
    parser.add_argument(
        "--trace-topk", type=int, default=d.trace_topk,
        help="how many top candidate tokens to capture per position per step (default 5)",
    )
    parser.add_argument(
        "--trace-positions", nargs="+", type=int, default=d.trace_positions,
        help="output positions to trace (default: the steered positions)",
    )
    parser.add_argument(
        "--trace-file", default=d.trace_file,
        help="if given, write the full per-step trace (tokens+probs) to this JSON file; "
             "implies --trace. If omitted, nothing is written to disk",
    )
    return parser


def run_experiment(args: "SteerConfig | argparse.Namespace", tokenizer) -> dict:
    """Run one steering experiment with an already-loaded `tokenizer`, printing the
    baseline, the steered result, and (if requested) the compact trajectory + trace file.

    `args` is any attribute bag with the CLI's field names -- a `SteerConfig` (typed,
    convenient for batch/programmatic use) or an argparse `Namespace` (the CLI path).
    Split out from `main` so batch runners can reuse a single tokenizer across many runs
    instead of reloading it each time. Returns the steered result dict.
    """
    # A filename implies tracing; otherwise there'd be nothing to write.
    trace = args.trace or args.trace_file is not None
    where = {"host": args.host, "port": args.port}

    # 1. Baseline: what the model says on its own.
    print("\nPROMPT:", args.prompt)
    base = steer(args.prompt, tokens=[], positions=[], seed=0, **where)
    print("baseline:", base["text"])

    # 2. Steer the target string(s) into place, tokenized + auto-positioned.
    n = len(args.target)
    modes = _per_target(args.mode, n, "mode")
    steps = _per_target(args.step, n, "step")
    pairs = ", ".join(
        f"{t!r}@pos{p}[{m} from step {s}]"
        for t, p, m, s in zip(args.target, args.start_pos, modes, steps)
    )
    print(f"\nSteering {pairs} (k={args.k}, p={args.prob}):")
    result = steer_strings(
        args.prompt, args.target, args.start_pos, tokenizer,
        probabilities=args.prob, ks=args.k, modes=args.mode, steps=args.step,
        trace=trace, trace_topk=args.trace_topk, trace_positions=args.trace_positions,
        **where,
    )
    # Show the FULL text plus exactly what landed at the pinned positions, so the steered
    # tokens are visible (not clipped) and you can see EOS only appears after them.
    landed = "".join(o["actual_token"] for o in result["interventions"])
    print("steered :", result["text"])
    print(f"  pinned positions {result['positions']} landed as: {landed!r}")
    print("  all_held:", result["all_held"])

    # 3. Per-step trace: print a compact top-1 trajectory to the terminal, and write the
    # FULL per-step top-k detail to the file the user named (the terminal stays readable;
    # the file is the artifact for tracking back through the model's reasoning).
    if trace and result.get("trace"):
        decoded = decode_trace(result["trace"], tokenizer)
        print_trace_summary(decoded)
        if args.trace_file is not None:
            write_trace_file(args.trace_file, {
                "prompt": args.prompt,
                "targets": args.target,
                "start_pos": args.start_pos,
                "mode": args.mode,
                "step": args.step,
                "text": result["text"],
                "interventions": result["interventions"],
                "positions": result["positions"],
                "token_ids": result["token_ids"],
                "trace": decoded,
            })

    return result


def main() -> None:
    args = build_parser().parse_args()
    print("Loading tokenizer (no GPU)...", flush=True)
    tokenizer = load_tokenizer()
    run_experiment(args, tokenizer)


if __name__ == "__main__":
    main()
