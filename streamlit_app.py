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
# Denoising film-strip: render the canvas at every step with confidence-driven
# opacity + blur, so "uncertain early, certain late" reads visually.
# ---------------------------------------------------------------------------

def _step_canvas_html(decoded: list[dict], step_idx: int, positions: list[int],
                      steered_positions: set[int]) -> str:
    """One step rendered as inline tokens.

    Each token's *opacity* is its top-1 probability and a *blur* is applied at low
    probability, so a step where the model is unsure reads as a hazy grey blur, and a
    step where it has committed reads as crisp black text. Steered (pinned) positions
    are tinted so you can see the intervention's reach.
    """
    rec = next((r for r in decoded if r["step_idx"] == step_idx), None)
    if rec is None:
        return "<em>(no record at this step)</em>"
    spans: list[str] = []
    for pos in positions:
        cands = rec["positions"].get(pos, [])
        if not cands:
            spans.append("<span style='opacity:.15;color:#999'>·</span>")
            continue
        tok = cands[0]["token"]
        prob = float(cands[0]["prob"])
        # Map probability to perceptual cues: low prob -> faint + blurred, high prob -> bold.
        # Floor opacity so faint tokens are still legible enough to recognize the canvas.
        opacity = 0.15 + 0.85 * prob
        blur_px = max(0.0, 3.0 * (1.0 - prob))
        weight = 700 if prob > 0.85 else 400
        color = "#1f4ed8" if pos in steered_positions else "#111"
        # Whitespace tokens would collapse in HTML; show them as a visible glyph.
        display = tok.replace(" ", "·").replace("\n", "⏎")
        spans.append(
            f"<span title='pos {pos} · p={prob:.2f}' "
            f"style='display:inline-block;margin:0 1px;padding:1px 3px;"
            f"opacity:{opacity:.3f};filter:blur({blur_px:.2f}px);"
            f"font-weight:{weight};color:{color};border-radius:3px;"
            f"background:{'#eef3ff' if pos in steered_positions else 'transparent'}'>"
            f"{display}</span>"
        )
    return "".join(spans)


# ---------------------------------------------------------------------------
# Targets editor: one row per intervention target.
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = pd.DataFrame(
    [{"target": "Yes", "start_pos": 0, "mode": "pin", "step": 0}]
)


def targets_editor() -> pd.DataFrame:
    """Editable table where each row is one (target string, start_pos, mode, step).

    Mirrors `SteerConfig`'s per-target parallel lists -- the table form keeps related
    fields together so you can't desync `start_pos[i]` from `target[i]`.

    The data_editor itself supports row removal (select a row's leftmost checkbox and
    press the trash icon in its toolbar), but that affordance is easy to miss, so we
    expose explicit "Clear all" / "Reset to default" buttons next to the table.
    """
    if "targets_df" not in st.session_state:
        st.session_state["targets_df"] = DEFAULT_TARGETS.copy()

    # Buttons mutate session_state *before* the editor renders so the editor picks up
    # the new value on this run. Each click sets a fresh editor key so Streamlit drops
    # any stale per-row UI state from the previous frame.
    btn_clear, btn_reset, _ = st.columns([1, 1, 6])
    if btn_clear.button("Clear all rows", help="empty the table"):
        st.session_state["targets_df"] = DEFAULT_TARGETS.iloc[0:0].copy()
        st.session_state["targets_editor_nonce"] = st.session_state.get("targets_editor_nonce", 0) + 1
    if btn_reset.button("Reset to default", help="restore the example row"):
        st.session_state["targets_df"] = DEFAULT_TARGETS.copy()
        st.session_state["targets_editor_nonce"] = st.session_state.get("targets_editor_nonce", 0) + 1

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
        key=f"targets_editor_{st.session_state.get('targets_editor_nonce', 0)}",
    )
    st.session_state["targets_df"] = edited
    st.caption(
        "Tip: to remove a row, click its left-edge checkbox and then the 🗑️ icon that "
        "appears in the table's toolbar (top-right of the table). Or use **Clear all rows** above."
    )
    return edited


# ---------------------------------------------------------------------------
# UI.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DiffusionGemma fill-attack workbench", layout="wide")
st.title("DiffusionGemma fill-attack workbench")

with st.sidebar:
    st.subheader("Server")
    st.caption(
        "ℹ️ The values below are the working defaults for the bundled `server.py`. "
        "**You don't need to change anything here** unless you're running the server "
        "on a different host/port."
    )
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

# --- Inputs (ordered: targets first, then prompt, then everything else) ---
st.markdown("### 1. Targets")
st.markdown(
    "Each row is one intervention -- the **target string** is the text you're forcing "
    "into the canvas, **start_pos** is the token position it lands at, **mode** picks "
    "between a hard pin and a one-shot perturbation, and **step** is the denoising "
    "step (0..~47) at which the intervention fires. Add rows for staggered steers."
)
targets_df = targets_editor()

st.markdown("### 2. Prompt")
prompt = st.text_area(
    "Prompt sent to the model",
    value="Is a hot dog a sandwich? Give a one-word verdict (Yes or No), then explain.",
    height=90,
    label_visibility="collapsed",
)

with st.expander("Advanced: all other inputs (sampling, trace, server)", expanded=False):
    st.markdown(
        "These are the same values shown in the sidebar -- mirrored here so every input "
        "is reachable from the main panel. Editing either copy syncs the run."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sampling**")
        st.write(f"k (top-k width): `{int(k)}`")
        st.write(f"prob (mass on target): `{float(prob)}`")
        st.write(f"seed: `{int(seed)}`")
    with c2:
        st.markdown("**Trace**")
        st.write(f"trace topk: `{int(trace_topk)}`")
        st.write(f"extra trace positions: `{extra_positions or '(none)'}`")
    with c3:
        st.markdown("**Server**")
        st.write(f"host: `{host}`")
        st.write(f"port: `{int(port)}`")

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
st.markdown(
    f"Steering acted on token positions **{last['positions']}**, and what actually "
    f"landed there in the final canvas was **`{last['landed']!r}`**. "
    f"&nbsp;·&nbsp; **{held_badge}** -- whether every pin survived denoising."
)
st.caption(
    "Read this as: *\"I asked for these tokens at these positions; here's what came out.\"* "
    "If `all_held` is ✅, the attack stuck verbatim; if ⚠️, the model overrode at least one pin."
)

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

# --- Denoising film-strip ---------------------------------------------------
# Renders every step as an inline canvas of tokens, where each token's opacity and
# blur reflect the model's top-1 probability at that position. Early steps -> blurry,
# faint, "unsure" looking; late steps -> crisp, bold, "committed" looking. The result
# is a visceral sense of the model steering itself toward an answer.
st.markdown("### Denoising film-strip -- watch the canvas sharpen step by step")
st.caption(
    "Each row is one denoising step. Token **opacity** = top-1 probability, **blur** "
    "fades as the model commits. Blue-tinted tokens are at steered (pinned) positions. "
    "Early rows look hazy because the model is still unsure; late rows look crisp because "
    "every position has locked onto a winner."
)
all_steps = sorted({rec["step_idx"] for rec in decoded})
all_positions = sorted({int(p) for rec in decoded for p in rec["positions"]})
steered_set = set(last["positions"])
if all_steps and all_positions:
    # Sample steps so the strip stays readable on long runs (~48 steps): show every
    # step up to ~24, otherwise stride. The user can also scrub to a specific step below.
    stride = max(1, len(all_steps) // 24)
    sampled = all_steps[::stride]
    if all_steps[-1] not in sampled:
        sampled.append(all_steps[-1])
    rows_html = []
    for s in sampled:
        canvas = _step_canvas_html(decoded, s, all_positions, steered_set)
        rows_html.append(
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"padding:4px 6px;border-bottom:1px solid #eee;font-family:monospace;font-size:14px'>"
            f"<div style='width:64px;color:#888;font-size:11px'>step {s:>3}</div>"
            f"<div>{canvas}</div></div>"
        )
    st.markdown(
        "<div style='border:1px solid #e3e3e3;border-radius:6px;padding:4px;"
        "background:#fafafa;max-height:520px;overflow-y:auto'>"
        + "".join(rows_html) +
        "</div>",
        unsafe_allow_html=True,
    )

    # Step scrubber: pick any step and see the full canvas at native size, no stride.
    st.markdown("**Scrub to a single step** (full canvas, no sampling)")
    pick = st.slider(
        "denoising step",
        min_value=int(all_steps[0]), max_value=int(all_steps[-1]),
        value=int(all_steps[0]), step=1,
    )
    full = _step_canvas_html(decoded, int(pick), all_positions, steered_set)
    st.markdown(
        f"<div style='border:1px solid #e3e3e3;border-radius:6px;padding:14px;"
        f"background:#fff;font-family:monospace;font-size:18px;line-height:1.8'>{full}</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("### Per-position telemetry")

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
