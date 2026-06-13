"""End-to-end smoke test for the steering library (needs the A100, loads the model).

The tmux `diffgemma` server holds a copy of the model, so free the GPU first:
    tmux kill-session -t diffgemma
    .venv/bin/python -m steering._smoke
    # then restart the server if you want it back:
    tmux new-session -d -s diffgemma '.venv/bin/python server.py 2>&1 | tee server.log'

Prints PASS/FAIL for each check in the plan's verification section.
"""

from __future__ import annotations

import torch

from load_model import load_model
from steering import RecorderConfig, make_intervention, resolve_token_id, steer

PROMPT = "Count up from one: 1, 2, 3,"
SEED = 0


def _ok(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def final_step_record(trace: list[dict]) -> dict:
    """The record at the last denoising step (cur_step == 1)."""
    return min(trace, key=lambda r: r["cur_step"])


def topk_at(record: dict, position: int):
    """Return (ids, probs) lists for `position` from a recorder record."""
    row = (record["positions"] == position).nonzero().flatten().item()
    return record["topk_ids"][row].tolist(), record["topk_probs"][row].tolist()


def main() -> None:
    print("Loading model...", flush=True)
    model, processor = load_model()
    canvas = model.config.canvas_length
    print(f"Loaded. canvas_length={canvas}\n", flush=True)

    # Pick a clearly single-token target. (This tokenizer has a standalone space
    # token, so " 7" is two tokens but bare "7" is one.)
    target_str = "7"
    target_id = resolve_token_id(target_str, processor)
    print(f"target {target_str!r} -> id {target_id}\n", flush=True)

    # --- Check 1: hard freeze (pin, k=1) at several positions -------------------------
    positions = [0, 3, 10]
    res = steer(
        model, processor, PROMPT,
        tokens=[target_id] * len(positions), positions=positions,
        ks=1, modes="pin", seed=SEED, max_new_tokens=canvas,
    )
    held = [o.held for o in res.interventions]
    print("Check 1 — hard freeze pin/k=1:")
    print(res.summary())
    print(f"  => {_ok(all(held))} (all positions hold the frozen token)\n", flush=True)

    # --- Check 2: probability accuracy (pin, k=2, p=0.95) via the trace ----------------
    p_target = 0.95
    res2 = steer(
        model, processor, PROMPT,
        tokens=[target_id], positions=[5], ks=2, probabilities=p_target, modes="pin",
        seed=SEED, max_new_tokens=canvas,
        record=RecorderConfig(positions=[5], top_k=5),
    )
    rec = final_step_record(res2.trace)
    ids, probs = topk_at(rec, position=5)
    top_is_target = ids[0] == target_id
    prob_close = abs(probs[0] - p_target) < 0.02
    print("Check 2 — probability accuracy pin/k=2/p=0.95:")
    print(f"  top id {ids[0]} (target {target_id}), prob {probs[0]:.4f}, runner {probs[1]:.4f}")
    print(f"  => {_ok(top_is_target and prob_close)} (effective prob ~= 0.95)\n", flush=True)

    # --- Check 3: proportional residual (k=3) — target ~p, two runners carry ~1-p ------
    res3 = steer(
        model, processor, PROMPT,
        tokens=[target_id], positions=[5], ks=3, probabilities=0.9, modes="pin",
        seed=SEED, max_new_tokens=canvas,
        record=RecorderConfig(positions=[5], top_k=5),
    )
    rec3 = final_step_record(res3.trace)
    ids3, probs3 = topk_at(rec3, position=5)
    residual = sum(probs3[1:3])
    print("Check 3 — k=3 residual split:")
    print(f"  top id {ids3[0]} prob {probs3[0]:.4f}; runners {probs3[1]:.4f}+{probs3[2]:.4f}={residual:.4f}")
    print(f"  => {_ok(ids3[0] == target_id and abs(probs3[0]-0.9) < 0.02 and abs(residual-0.1) < 0.02)}\n", flush=True)

    # --- Check 4: perturb vs pin at an early step -------------------------------------
    early = 2
    common = dict(tokens=[target_id], positions=[7], ks=1, seed=SEED, max_new_tokens=canvas, steps=early)
    pin_res = steer(model, processor, PROMPT, modes="pin", **common)
    perturb_res = steer(model, processor, PROMPT, modes="perturb", **common)
    print("Check 4 — perturb vs pin (early step):")
    print(f"  pin    held={pin_res.interventions[0].held} got={pin_res.interventions[0].actual_token!r}")
    print(f"  perturb held={perturb_res.interventions[0].held} got={perturb_res.interventions[0].actual_token!r}")
    print(f"  => {_ok(pin_res.interventions[0].held)} (pin guarantees output; perturb may differ)\n", flush=True)

    # --- Check 5: recorder shapes over the full canvas --------------------------------
    res5 = steer(
        model, processor, PROMPT,
        tokens=[target_id], positions=[0], ks=1, modes="pin", seed=SEED, max_new_tokens=canvas,
        record=True,  # default: all positions, top_k=50, argmax + entropy
    )
    r0 = res5.trace[0]
    shapes_ok = (
        r0["argmax"].shape == (canvas,)
        and r0["entropy"].shape == (canvas,)
        and r0["topk_ids"].shape == (canvas, 50)
    )
    print("Check 5 — recorder shapes:")
    print(f"  steps recorded: {len(res5.trace)}; argmax {tuple(r0['argmax'].shape)}, "
          f"entropy {tuple(r0['entropy'].shape)}, topk_ids {tuple(r0['topk_ids'].shape)}")
    print(f"  => {_ok(shapes_ok)}\n", flush=True)


if __name__ == "__main__":
    main()
