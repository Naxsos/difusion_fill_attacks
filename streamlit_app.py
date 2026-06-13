"""Streamlit workbench for the diffusion fill-attack experiments.

A frontend over the same `client.steer` + `example_steer.steer_strings` pipeline that
`run_experiments.py` and the CLI use, so anything you can do here you can also do from
the terminal -- nothing about the model or server changes. The heavy ~52 GB model stays
on whichever box runs `server.py`; this app only needs `streamlit`, `pandas`, and the
lightweight tokenizer (loaded once and cached).

What the workbench gives you
----------------------------
1. A form for every `SteerConfig` knob (prompt, per-target text/start_pos/mode/step,
   k/prob, seed, server host/port, trace topk/positions). Multi-target steering uses
   one row per target so you can stage interventions at different denoising steps.
2. A run button that calls the same `steer_strings(...)` the CLI uses, prints the
   baseline next to the steered text, lists what landed at each pinned position, and
   surfaces `all_held`.
3. A convergence visualization for DiffusionGemma's denoising loop, built from the
   per-step trace that the server records at the traced positions:
     * top-1 token trajectory per position (a heatmap-style table -- one column per
       denoising step, one row per traced position),
     * top-1 probability curves over denoising steps per position (line chart),
     * top-k probability stack at a selected position (stacked area chart, so you can
       see the model commit to a token as competing tokens decay).

Launch (the diffusion server must already be running -- see SERVER.md):

    pip install streamlit pandas
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import traceback

import pandas as pd
import streamlit as st

from client import steer as steer_call
from example_steer import decode_trace, load_tokenizer, steer_strings


# Cache the tokenizer across reruns. Streamlit reruns the script top-to-bottom on every
# interaction; without this we'd re-download/parse the tokenizer each click. The model is
# remote (server.py), so this only loads tokenizer files -- still worth caching.
@st.cache_resource(show_spinner="Loading tokenizer (no GPU)...")
def _tokenizer():
    return load_tokenizer()


# ---------------------------------------------------------------------------
# Trace -> chartable frames.
# ---------------------------------------------------------------------------

def trajectory_frame(decoded: list[dict]) -> pd.DataFrame:
    """One row per traced position, one column per denoising step, value = top-1 token.

    Reads the same `decoded` shape `print_trace_summary` uses, so what you see here
    matches what the CLI prints. Steps may be missing for some records (the recorder
    only fires when a position is touched), so we fill gaps with empty strings.
    """
    rows: dict[int, dict[int, str]] = {}
    for rec in decoded:
        step = rec["step_idx"]
        for pos, cands in rec["positions"].items():
            if cands:
                rows.setdefault(int(pos), {})[step] = cands[0]["token"]
    if not rows:
        return pd.DataFrame()
    all_steps = sorted({s for r in rows.values() for s in r})
    df = pd.DataFrame(
        {step: [rows[p].get(step, "") for p in sorted(rows)] for step in all_steps},
        index=[f"pos {p}" for p in sorted(rows)],
    )
    df.columns = [f"step {s}" for s in df.columns]
    return df


def top1_prob_frame(decoded: list[dict]) -> pd.DataFrame:
    """Top-1 probability over denoising steps, one column per traced position.

    Lets you see how confident the model is at each pinned/traced position step by
    step -- a hard pin reads as a flat 1.0; a "perturb"-then-release reads as a spike
    that decays; an unsteered position reads as a slow climb to commitment.
    """
    rows: list[dict] = []
    for rec in decoded:
        row = {"step": rec["step_idx"]}
        for pos, cands in rec["positions"].items():
            if cands:
                row[f"pos {pos}"] = cands[0]["prob"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("step").last().sort_index()
    return df


def topk_at_position_frame(decoded: list[dict], position: int, top_k: int) -> pd.DataFrame:
    """Top-k token probabilities at one position over denoising steps.

    The point is to watch the *competing* candidates collapse as the canvas commits:
    early steps spread mass over many tokens, late steps put almost all of it on one.
    Tokens that never enter the top-k are dropped to keep the legend readable.
    """
    rows: list[dict] = []
    for rec in decoded:
        cands = rec["positions"].get(position, [])
        if not cands:
            continue
        row = {"step": rec["step_idx"]}
        for c in cands[:top_k]:
            row[repr(c["token"])] = c["prob"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("step").last().sort_index().fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Targets editor: one row per intervention target.
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = pd.DataFrame(
    [{"target": " 9, 8, 7", "start_pos": 0, "mode": "pin", "step": 0}]
)


def targets_editor() -> pd.DataFrame:
    """Editable table where each row is one (target string, start_pos, mode, step).

    Mirrors `SteerConfig`'s per-target parallel lists -- the table form keeps related
    fields together so you can't desync `start_pos[i]` from `target[i]`.
    """
    if "targets_df" not in st.session_state:
        st.session_state["targets_df"] = DEFAULT_TARGETS.copy()
    edited = st.data_editor(
        st.session_state["targets_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "target": st.column_config.TextColumn(
                "target string", help="will be tokenized and pinned to consecutive positions"
            ),
            "start_pos": st.column_config.NumberColumn("start_pos", min_value=0, step=1),
            "mode": st.column_config.SelectboxColumn("mode", options=["pin", "perturb"]),
            "step": st.column_config.NumberColumn(
                "step", min_value=0, step=1,
                help="denoising step (0..~47) at which this target fires",
            ),
        },
        key="targets_editor",
    )
    st.session_state["targets_df"] = edited
    return edited


# ---------------------------------------------------------------------------
# UI.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DiffusionGemma fill-attack workbench", layout="wide")
st.title("DiffusionGemma fill-attack workbench")
st.caption(
    "Frontend over `client.steer` -- the heavy model stays on the server. "
    "Equivalent to running `example_steer.py` / `run_experiments.py`, with a convergence view."
)

with st.sidebar:
    st.subheader("Server")
    host = st.text_input("host", value="localhost")
    port = st.number_input("port", value=8000, step=1)
    st.subheader("Sampling")
    k = st.number_input(
        "k (top-k width)", min_value=1, value=1, step=1,
        help="1 = hard freeze on the target; >=2 spreads residual mass across runner-ups",
    )
    prob = st.number_input(
        "prob (mass on target; 0 = hard pin)", min_value=0.0, max_value=1.0,
        value=0.0, step=0.05,
        help="0 means leave probabilities unset (hard pin). >0 sets per-token mass.",
    )
    seed = st.number_input("seed", value=0, step=1)
    st.subheader("Trace")
    trace_topk = st.number_input("trace topk", min_value=1, value=5, step=1)
    extra_positions = st.text_input(
        "extra trace positions (comma-separated, optional)", value="",
        help="defaults to the steered positions; add more here to track unsteered tokens",
    )

prompt = st.text_area(
    "Prompt",
    value="Is a hot dog a sandwich? Give a one-word verdict (Yes or No), then explain.",
    height=90,
)

st.markdown("**Targets** -- each row is one intervention. Add rows for staggered steers.")
targets_df = targets_editor()

run = st.button("Run experiment", type="primary")

# Persist the last result across reruns (Streamlit reruns on widget change), so the
# convergence view can be tweaked (which position to plot, top-k width) without rerunning
# the model -- the trace is already in memory.
if run:
    df = targets_df.dropna(subset=["target"]).copy()
    df = df[df["target"].astype(str).str.len() > 0]
    if df.empty:
        st.error("Add at least one target row.")
        st.stop()

    targets = df["target"].astype(str).tolist()
    start_pos = df["start_pos"].astype(int).tolist()
    modes = df["mode"].astype(str).tolist()
    steps = df["step"].astype(int).tolist()

    extra: list[int] = []
    if extra_positions.strip():
        try:
            extra = [int(x) for x in extra_positions.split(",") if x.strip()]
        except ValueError:
            st.error("Extra trace positions must be integers separated by commas.")
            st.stop()

    where = {"host": host, "port": int(port)}
    tokenizer = _tokenizer()

    with st.spinner("Calling server..."):
        try:
            base = steer_call(prompt, tokens=[], positions=[], seed=int(seed), **where)
            # Pre-compute the trace_positions list so unsteered positions can be tracked too.
            steered_positions: list[int] = []
            for tgt, sp in zip(targets, start_pos):
                ids = tokenizer.encode(tgt, add_special_tokens=False)
                steered_positions.extend(range(sp, sp + len(ids)))
            trace_positions = sorted(set(steered_positions) | set(extra))

            result = steer_strings(
                prompt, targets, start_pos, tokenizer,
                probabilities=(prob if prob > 0 else None),
                ks=int(k),
                modes=modes, steps=steps,
                trace=True, trace_topk=int(trace_topk),
                trace_positions=trace_positions,
                seed=int(seed),
                **where,
            )
        except Exception as exc:  # noqa: BLE001 - surface the server error to the user
            st.error(f"Server call failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()

    decoded = decode_trace(result.get("trace", []), tokenizer) if result.get("trace") else []
    landed = "".join(o["actual_token"] for o in result["interventions"])
    st.session_state["last_run"] = {
        "prompt": prompt,
        "baseline": base["text"],
        "steered": result["text"],
        "landed": landed,
        "positions": result["positions"],
        "all_held": result["all_held"],
        "interventions": result["interventions"],
        "decoded": decoded,
        "trace_positions": trace_positions,
        "config": {
            "targets": targets, "start_pos": start_pos,
            "modes": modes, "steps": steps,
            "k": int(k), "prob": prob, "seed": int(seed),
        },
    }

last = st.session_state.get("last_run")
if last is None:
    st.info("Configure the prompt + targets and click **Run experiment**.")
    st.stop()

# --- Output ----------------------------------------------------------------
st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Baseline (no steering)")
    st.write(last["baseline"])
with right:
    st.subheader("Steered")
    st.write(last["steered"])

st.subheader("Pinned tokens")
held_badge = "✅ all_held" if last["all_held"] else "⚠️ NOT all held"
st.markdown(f"`positions={last['positions']}` -> landed as `{last['landed']!r}` &nbsp;·&nbsp; **{held_badge}**")

with st.expander("Raw interventions"):
    st.dataframe(pd.DataFrame(last["interventions"]), use_container_width=True)

# --- Convergence visualization --------------------------------------------
decoded = last["decoded"]
if not decoded:
    st.info("No trace was recorded -- nothing to visualize.")
    st.stop()

st.divider()
st.header("Convergence of the denoising loop")
st.caption(
    "DiffusionGemma denoises the whole canvas jointly over ~48 steps. These charts show "
    "what the sampler saw at each traced position on each step -- so you can watch a "
    "pin commit instantly and the surrounding tokens collapse onto a final answer."
)

traj = trajectory_frame(decoded)
st.markdown("**Top-1 token trajectory per traced position** (one column per denoising step)")
st.dataframe(traj, use_container_width=True)

probs = top1_prob_frame(decoded)
st.markdown("**Top-1 probability over denoising steps** (per traced position)")
st.line_chart(probs, height=320)

st.markdown("**Top-k probability stack at one position** -- watch competing tokens decay")
positions_seen = sorted({int(p) for rec in decoded for p in rec["positions"]})
focus = st.selectbox(
    "position to inspect", positions_seen,
    index=0 if positions_seen else None,
    format_func=lambda p: f"pos {p}" + ("  (steered)" if p in last["positions"] else ""),
)
focus_topk = st.slider("top-k width", min_value=2, max_value=int(trace_topk), value=min(5, int(trace_topk)))
if focus is not None:
    topk_df = topk_at_position_frame(decoded, int(focus), focus_topk)
    if topk_df.empty:
        st.info(f"No trace at position {focus}.")
    else:
        st.area_chart(topk_df, height=320)

# --- Export ---------------------------------------------------------------
st.divider()
payload = {
    "prompt": last["prompt"],
    "config": last["config"],
    "baseline": last["baseline"],
    "steered": last["steered"],
    "landed": last["landed"],
    "positions": last["positions"],
    "all_held": last["all_held"],
    "interventions": last["interventions"],
    "trace_positions": last["trace_positions"],
    "trace": decoded,
}
st.download_button(
    "Download run as JSON",
    data=json.dumps(payload, indent=2),
    file_name="streamlit_run.json",
    mime="application/json",
)
