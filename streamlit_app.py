"""Streamlit workbench for the diffusion fill-attack experiments — simple form UI.

The input UI is a flat form: one widget per CLI flag emitted by
``steer_config.to_cli``. The Results and Convergence views are unchanged from
the previous workbench.

Launch (the diffusion server must already be running -- see SERVER.md):

    pip install streamlit pandas altair
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
import traceback

import altair as alt
import pandas as pd
import streamlit as st

from client import steer as steer_call
from example_steer import decode_trace, load_tokenizer, steer_strings
from steer_config import SteerConfig, to_cli


# Cache the tokenizer across reruns. Streamlit reruns the script top-to-bottom on every
# interaction; without this we'd re-parse the tokenizer each click.
@st.cache_resource(show_spinner="Loading tokenizer (no GPU)...")
def _tokenizer():
    return load_tokenizer()


# ---------------------------------------------------------------------------
# Trace -> chartable frames.
# ---------------------------------------------------------------------------

def trajectory_frame(decoded: list[dict]) -> pd.DataFrame:
    """One row per traced position, one column per denoising step, value = top-1 token."""
    rows: dict[int, dict[int, str]] = {}
    for rec in decoded:
        step = rec["step_idx"]
        for pos, cands in rec["positions"].items():
            if cands:
                rows.setdefault(int(pos), {})[step] = cands[0]["token"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        {step: [rows[p].get(step, "") for p in sorted(rows)]
         for step in sorted({s for r in rows.values() for s in r})},
        index=[f"pos {p}" for p in sorted(rows)],
    )
    df.columns = [f"step {s}" for s in df.columns]
    return df


def top1_prob_frame(decoded: list[dict]) -> pd.DataFrame:
    """Top-1 probability over denoising steps, one column per traced position.

    Uses pre-intervention (natural) probs when available so the curve reflects the
    model's genuine confidence rather than the forced distribution.
    """
    rows: list[dict] = []
    for rec in decoded:
        row = {"step": rec["step_idx"]}
        source = rec.get("pre_positions") or rec["positions"]
        for pos, cands in source.items():
            if cands:
                row[f"pos {pos}"] = cands[0]["prob"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("step").last().sort_index()


def topk_at_position_frame(decoded: list[dict], position: int, top_k: int) -> pd.DataFrame:
    """Top-k token probabilities at one position over denoising steps (long format)."""
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
    return pd.DataFrame(rows).groupby("step").last().sort_index().fillna(0.0)


def distribution_at(decoded: list[dict], step: int, position: int, top_k: int) -> pd.DataFrame:
    """Top-k token candidates at (step, position), sorted by probability descending.

    Returns post-intervention probabilities.  For steered positions, a companion
    ``pre_distribution_at`` call gives the natural (pre-intervention) distribution.
    """
    rec = next((r for r in decoded if r["step_idx"] == step), None)
    if rec is None:
        return pd.DataFrame()
    cands = rec["positions"].get(position, [])
    if not cands:
        return pd.DataFrame()
    df = pd.DataFrame(cands[:top_k])
    df["display"] = df["token"].apply(lambda t: t.replace(" ", "·").replace("\n", "⏎") or "∅")
    return df[["display", "token", "prob"]].sort_values("prob", ascending=False).reset_index(drop=True)


def pre_distribution_at(decoded: list[dict], step: int, position: int, top_k: int) -> pd.DataFrame:
    """Top-k natural (pre-intervention) candidates at (step, position).

    Returns an empty DataFrame when no pre-intervention trace is available.
    """
    rec = next((r for r in decoded if r["step_idx"] == step), None)
    if rec is None:
        return pd.DataFrame()
    pre_pos = rec.get("pre_positions")
    if pre_pos is None:
        return pd.DataFrame()
    cands = pre_pos.get(position, [])
    if not cands:
        return pd.DataFrame()
    df = pd.DataFrame(cands[:top_k])
    df["display"] = df["token"].apply(lambda t: t.replace(" ", "·").replace("\n", "⏎") or "∅")
    return df[["display", "token", "prob"]].sort_values("prob", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streaming-process diagnostics.
# ---------------------------------------------------------------------------

def _entropy_of(cands: list[dict]) -> float:
    return -sum(c["prob"] * math.log(max(c["prob"], 1e-12)) for c in cands)


def entropy_frame(decoded: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in decoded:
        for pos, cands in rec["positions"].items():
            if cands:
                rows.append({"step": rec["step_idx"], "position": int(pos),
                             "entropy": _entropy_of(cands)})
    return pd.DataFrame(rows)


def mean_entropy_curve(decoded: list[dict]) -> pd.DataFrame:
    by_step: dict[int, list[float]] = {}
    for rec in decoded:
        ents = [_entropy_of(c) for c in rec["positions"].values() if c]
        if ents:
            by_step.setdefault(rec["step_idx"], []).extend(ents)
    return pd.DataFrame([
        {"step": s, "mean_entropy": sum(es) / len(es), "max_entropy": max(es)}
        for s, es in sorted(by_step.items())
    ])


def commitment_frame(decoded: list[dict], threshold: float = 0.9) -> pd.DataFrame:
    first: dict[int, int | None] = {}
    final_tok: dict[int, str] = {}
    final_prob: dict[int, float] = {}
    for rec in sorted(decoded, key=lambda r: r["step_idx"]):
        for pos, cands in rec["positions"].items():
            if not cands:
                continue
            pos = int(pos)
            top = cands[0]
            if first.get(pos) is None and top["prob"] >= threshold:
                first[pos] = rec["step_idx"]
            final_tok[pos] = top["token"]
            final_prob[pos] = top["prob"]
    rows = []
    for pos in sorted(final_tok):
        rows.append({
            "position": pos,
            "commit_step": first.get(pos),
            "final_token": final_tok[pos].replace(" ", "·").replace("\n", "⏎") or "∅",
            "final_prob": final_prob[pos],
        })
    return pd.DataFrame(rows)


def final_rank_frame(decoded: list[dict]) -> pd.DataFrame:
    by_step = sorted(decoded, key=lambda r: r["step_idx"])
    if not by_step:
        return pd.DataFrame()
    last = by_step[-1]
    final_winner: dict[int, str] = {
        int(p): cands[0]["token"]
        for p, cands in last["positions"].items() if cands
    }
    rows = []
    for rec in by_step:
        for pos, cands in rec["positions"].items():
            if not cands or int(pos) not in final_winner:
                continue
            target = final_winner[int(pos)]
            rank = next((i + 1 for i, c in enumerate(cands) if c["token"] == target), len(cands) + 1)
            rows.append({"step": rec["step_idx"], "position": int(pos), "rank": rank})
    return pd.DataFrame(rows)


def churn_frame(decoded: list[dict]) -> pd.DataFrame:
    seen: dict[int, set[str]] = {}
    for rec in decoded:
        for pos, cands in rec["positions"].items():
            if cands:
                seen.setdefault(int(pos), set()).add(cands[0]["token"])
    return pd.DataFrame([
        {"position": p, "distinct_top1": len(s)}
        for p, s in sorted(seen.items())
    ])


# ---------------------------------------------------------------------------
# Step canvas: render every traced position at one step with confidence-driven
# opacity + blur, so "uncertain early, certain late" reads visually.
# ---------------------------------------------------------------------------

def _step_canvas_html(decoded: list[dict], step_idx: int, positions: list[int],
                      steered_positions: set[int], focus: int | None = None) -> str:
    rec = next((r for r in decoded if r["step_idx"] == step_idx), None)
    if rec is None:
        return "<em>(no record at this step)</em>"

    # Positions actively steered *at this specific step*.
    active_steered: set[int] = set(rec.get("steered_positions", []))
    pre_pos = rec.get("pre_positions")  # natural distribution before intervention

    spans: list[str] = []
    for pos in positions:
        cands = rec["positions"].get(pos, [])
        if not cands:
            spans.append("<span style='opacity:.15;color:#999'>·</span>")
            continue

        tok = cands[0]["token"]
        post_prob = float(cands[0]["prob"])

        # Opacity/blur encode the *natural* model confidence when available.
        if pre_pos is not None:
            pre_cands = pre_pos.get(pos, [])
            nat_prob = float(pre_cands[0]["prob"]) if pre_cands else post_prob
        else:
            nat_prob = post_prob

        opacity = 0.15 + 0.85 * nat_prob
        blur_px = max(0.0, 3.0 * (1.0 - nat_prob))
        weight = 700 if nat_prob > 0.85 else 400

        is_active = pos in active_steered
        if pos == focus:
            color, bg, border = "#b91c1c", "#fee2e2", "1px solid #ef4444"
        elif is_active:
            # Green dashed border = steered right now; blue fill = steered position overall.
            color, bg, border = "#166534", "#dcfce7", "2px dashed #16a34a"
        elif pos in steered_positions:
            color, bg, border = "#1f4ed8", "#eef3ff", "0"
        else:
            color, bg, border = "#111", "transparent", "0"

        display = tok.replace(" ", "·").replace("\n", "⏎") or "∅"

        # Tooltip: show natural prob + post-intervention prob when they differ.
        if pre_pos is not None and is_active:
            pre_cands = pre_pos.get(pos, [])
            nat_tok = pre_cands[0]["token"] if pre_cands else "?"
            nat_p = float(pre_cands[0]["prob"]) if pre_cands else 0.0
            title = (
                f"pos {pos} | natural: '{nat_tok.replace(chr(39), '`')}' p={nat_p:.2f}"
                f" → steered: p={post_prob:.2f}"
            )
        else:
            title = f"pos {pos} · p={nat_prob:.2f}"

        spans.append(
            f"<span title='{title}' "
            f"style='display:inline-block;margin:0 1px;padding:1px 3px;"
            f"opacity:{opacity:.3f};filter:blur({blur_px:.2f}px);"
            f"font-weight:{weight};color:{color};border:{border};border-radius:3px;"
            f"background:{bg}'>{display}</span>"
        )
    return "".join(spans)


# ---------------------------------------------------------------------------
# Dynamic targets state. One row = one --target with its parallel start_pos /
# step / prob. Add a row with the per-row "+" button; remove with "🗑".
# ---------------------------------------------------------------------------

DEFAULT_ROW = {"target": "Yes", "start_pos": 0, "step": 0, "prob": 0.0}


def _ensure_rows() -> None:
    if "rows" not in st.session_state:
        st.session_state["rows"] = [dict(DEFAULT_ROW)]
    # Bumped when we mutate the row list, so widget keys move and old per-row
    # state can't bleed into a renumbered row.
    if "rows_nonce" not in st.session_state:
        st.session_state["rows_nonce"] = 0


def _add_row_cb() -> None:
    rows = st.session_state["rows"]
    template = dict(rows[-1]) if rows else dict(DEFAULT_ROW)
    rows.append(template)
    st.session_state["rows_nonce"] += 1


def _remove_row_cb(idx: int) -> None:
    rows = st.session_state["rows"]
    if 0 <= idx < len(rows):
        rows.pop(idx)
    st.session_state["rows_nonce"] += 1


def _persist_row(idx: int, nonce: int) -> None:
    """Pull current widget values for row `idx` back into st.session_state['rows']."""
    row = st.session_state["rows"][idx]
    row["target"] = st.session_state.get(f"row_target_{nonce}_{idx}", row["target"])
    row["start_pos"] = int(st.session_state.get(f"row_sp_{nonce}_{idx}", row["start_pos"]))
    row["step"] = int(st.session_state.get(f"row_step_{nonce}_{idx}", row["step"]))
    row["prob"] = float(st.session_state.get(f"row_prob_{nonce}_{idx}", row["prob"]))


def _parse_ints(text: str) -> list[int]:
    """Space-or-comma-separated ints. Empty string -> []."""
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DiffusionGemma fill-attack workbench", layout="wide")

# Tighten the default Streamlit chrome so the form starts near the top of the
# viewport rather than wasting ~80px of header padding.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem !important; padding-bottom: 2rem !important;}
      header[data-testid="stHeader"] {height: 2.5rem;}
      h1, h2, h3, h4 {margin-top: 0.2rem !important; margin-bottom: 0.4rem !important;}
      div[data-testid="stTabs"] {margin-top: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

defaults = SteerConfig()
_ensure_rows()
# Persistent prompt -- bound to a session_state key so it survives tab switches
# and uploads can write into it.
st.session_state.setdefault("prompt_text", defaults.prompt)
# Which tab is showing. Set to "Results" right after a successful run so the
# main view auto-switches; reset by user clicks on the radio.
st.session_state.setdefault("active_tab", "Setup")


def _load_experiment_into_state(payload: dict) -> tuple[bool, str]:
    """Mirror an uploaded Simon's-style experiment JSON into session_state.

    Schema (extra fields are ignored):
      {prompt, targets, start_pos, [modes], [steps]}
    """
    if not isinstance(payload, dict):
        return False, "top-level JSON must be an object"
    prompt_v = payload.get("prompt")
    targets_v = payload.get("targets")
    start_pos_v = payload.get("start_pos")
    if not isinstance(prompt_v, str) or not prompt_v.strip():
        return False, "missing or empty `prompt`"
    if not isinstance(targets_v, list) or not targets_v:
        return False, "missing or empty `targets` array"
    if not isinstance(start_pos_v, list) or len(start_pos_v) != len(targets_v):
        return False, "`start_pos` must be a list the same length as `targets`"

    steps_v = payload.get("steps") or [0] * len(targets_v)
    if len(steps_v) != len(targets_v):
        return False, "`steps`, if provided, must match `targets` length"

    new_rows = []
    for tgt, sp, st_ in zip(targets_v, start_pos_v, steps_v):
        new_rows.append({
            "target": str(tgt),
            "start_pos": int(sp),
            "step": int(st_),
            "prob": 0.0,
        })
    st.session_state["prompt_text"] = prompt_v
    st.session_state["rows"] = new_rows
    st.session_state["rows_nonce"] += 1
    return True, f"loaded {len(new_rows)} target(s) at positions {list(start_pos_v)}"


# --- Sidebar (collapsed-by-default panels) ---------------------------------
with st.sidebar:
    st.markdown("### ▶ Run")
    submitted = st.button(
        "Run experiment", type="primary", use_container_width=True,
    )
    if "last_run" in st.session_state:
        st.caption(
            f"Last run: **{len(st.session_state['last_run'].get('decoded', []))}** trace records"
        )

    with st.expander("Server", expanded=False):
        st.caption("Defaults match the bundled `server.py` -- only change if your server moved.")
        host = st.text_input("--host", value=defaults.host)
        port = st.number_input("--port", value=defaults.port, step=1)

    with st.expander("Advanced", expanded=False):
        st.caption("Sampling, mode, tracing, reproducibility.")
        k = st.number_input(
            "--k  (top-k width; 1 = hard freeze)",
            min_value=1, value=defaults.k, step=1,
        )
        mode = st.selectbox(
            "--mode  (applies to every target)",
            options=["pin", "perturb"], index=0,
            help="`pin` = hard freeze through the end; `perturb` = one-shot nudge then release",
        )
        seed = st.number_input("seed", value=0, step=1)
        trace = st.checkbox(
            "--trace  (record per-step top-k for the convergence view)", value=True,
        )
        trace_topk = st.number_input(
            "trace topk", min_value=1, value=defaults.trace_topk, step=1,
            help="server-side per-step top-k width when --trace is on",
        )
        trace_positions_text = st.text_input(
            "--trace-positions  (optional; defaults to the steered positions)",
            value="", help="space-separated extra canvas positions to record",
        )

# --- Top tab nav (radio styled as tabs; programmatic switch on run) -------
TABS = ["Setup", "Results", "Convergence"]
# A successful run sets _pending_tab; we consume it here, before the widget
# is instantiated, so we never write to the widget's own key after creation.
if "_pending_tab" in st.session_state:
    st.session_state["active_tab"] = st.session_state.pop("_pending_tab")
active_tab = st.radio(
    "view", TABS,
    index=TABS.index(st.session_state.get("active_tab", "Setup")),
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)

# Pull rows + nonce -- both sit in session_state and are always available.
rows = st.session_state["rows"]
nonce = st.session_state["rows_nonce"]

# --- SETUP TAB -------------------------------------------------------------
if active_tab == "Setup":
    with st.expander("📂 Load experiment from JSON", expanded=False):
        st.caption(
            "Upload a Simon's-style experiment JSON (e.g. files in "
            "`simons_experiments/`); the **prompt** and **targets** below are filled in."
        )
        uploaded = st.file_uploader(
            "experiment file", type=["json"], label_visibility="collapsed",
            key=f"exp_uploader_{st.session_state.get('uploader_nonce', 0)}",
        )
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.read().decode("utf-8"))
                ok, msg = _load_experiment_into_state(payload)
                if ok:
                    st.session_state["uploader_nonce"] = (
                        st.session_state.get("uploader_nonce", 0) + 1
                    )
                    st.success(f"✅ {uploaded.name} -- {msg}")
                    st.rerun()
                else:
                    st.error(f"⚠ {uploaded.name}: {msg}")
            except json.JSONDecodeError as e:
                st.error(f"could not parse JSON: {e}")

    st.markdown("#### Prompt")
    st.text_area(
        "--prompt", height=90, label_visibility="collapsed", key="prompt_text",
    )

    st.markdown("#### Targets")
    st.caption(
        "One row per `--target`. Click ➕ on the last row to add another target with "
        "its own `--start-pos`, `--step`, and per-token `--prob`. Click 🗑 to remove a row."
    )

    if not rows:
        st.info("No targets. (Refresh the page to restore the default row.)")

    for i, row in enumerate(rows):
        is_last = i == len(rows) - 1
        c_t, c_sp, c_st, c_pr, c_add, c_del = st.columns(
            [4, 1.1, 1.1, 1.1, 0.6, 0.6], vertical_alignment="bottom",
        )
        c_t.text_area(
            f"target #{i + 1}",
            value=row["target"], height=68,
            key=f"row_target_{nonce}_{i}",
            on_change=_persist_row, args=(i, nonce),
        )
        c_sp.number_input(
            "start_pos", min_value=0, step=1, value=int(row["start_pos"]),
            key=f"row_sp_{nonce}_{i}",
            on_change=_persist_row, args=(i, nonce),
        )
        c_st.number_input(
            "step", min_value=0, step=1, value=int(row["step"]),
            key=f"row_step_{nonce}_{i}",
            on_change=_persist_row, args=(i, nonce),
        )
        c_pr.number_input(
            "prob", min_value=0.0, max_value=1.0, value=float(row["prob"]), step=0.05,
            key=f"row_prob_{nonce}_{i}",
            on_change=_persist_row, args=(i, nonce),
            help="0 = hard pin",
        )
        if is_last:
            c_add.button(
                "➕", key=f"row_add_{nonce}_{i}", help="add another target",
                on_click=_add_row_cb, use_container_width=True,
            )
        else:
            c_add.markdown("&nbsp;", unsafe_allow_html=True)
        if len(rows) > 1:
            c_del.button(
                "🗑", key=f"row_del_{nonce}_{i}", help="remove this target",
                on_click=_remove_row_cb, args=(i,), use_container_width=True,
            )
        else:
            c_del.markdown("&nbsp;", unsafe_allow_html=True)
        # Mirror current widget values into the row dict so the run picks them up
        # even if the user never blurred a field.
        _persist_row(i, nonce)

# --- Build cfg (used for CLI preview + run) -------------------------------
prompt = st.session_state["prompt_text"]
trace_positions = (
    _parse_ints(trace_positions_text) if trace_positions_text.strip() else None
)
targets = [r["target"] for r in rows if r["target"]]
start_pos = [int(r["start_pos"]) for r in rows if r["target"]]
steps = [int(r["step"]) for r in rows if r["target"]]
probs_per_target = [float(r["prob"]) for r in rows if r["target"]]
modes = [mode] * len(targets) if targets else list(defaults.mode)

# CLI preview surfaces a single --prob (the first row's). The actual run still
# passes per-target probabilities through to the server.
preview_prob = probs_per_target[0] if probs_per_target and probs_per_target[0] > 0 else None
cfg = SteerConfig(
    prompt=prompt,
    target=targets or list(defaults.target),
    start_pos=start_pos or list(defaults.start_pos),
    prob=preview_prob,
    k=int(k),
    mode=modes,
    step=steps or list(defaults.step),
    trace=bool(trace),
    trace_topk=int(trace_topk),
    trace_positions=trace_positions,
    host=host,
    port=int(port),
)

if active_tab == "Setup":
    with st.expander("equivalent CLI command", expanded=False):
        st.code(to_cli(cfg), language="bash")
        if len({p for p in probs_per_target if p > 0}) > 1:
            st.caption(
                "ℹ The CLI flag `--prob` is scalar; the run will still pass per-target "
                f"probs `{probs_per_target}` to the server."
            )

# ---------------------------------------------------------------------------
# Run. Triggered by the sidebar button regardless of the active tab.
# ---------------------------------------------------------------------------

if submitted:
    if not targets:
        st.error("Need at least one target row with a non-empty target string.")
        st.stop()

    tokenizer = _tokenizer()
    where = {"host": host, "port": int(port)}

    # Per-target prob expanded to per-token list (steer_strings's `probabilities`
    # accepts None / scalar / per-token list).
    if any(p > 0 for p in probs_per_target):
        per_token_probs: list[float] = []
        for tgt, p in zip(targets, probs_per_target):
            ids = tokenizer.encode(tgt, add_special_tokens=False)
            per_token_probs.extend([p if p > 0 else 0.0] * len(ids))
        probabilities_arg: list[float] | None = per_token_probs
    else:
        probabilities_arg = None

    with st.spinner("Calling server..."):
        try:
            base = steer_call(
                prompt, tokens=[], positions=[], seed=int(seed), **where
            )
            steered_positions: list[int] = []
            for tgt, sp in zip(targets, start_pos):
                ids = tokenizer.encode(tgt, add_special_tokens=False)
                steered_positions.extend(range(sp, sp + len(ids)))
            tp = sorted(set(steered_positions) | set(trace_positions or []))

            result = steer_strings(
                prompt, targets, start_pos, tokenizer,
                probabilities=probabilities_arg,
                ks=int(k),
                modes=modes, steps=steps,
                trace=bool(trace), trace_topk=int(trace_topk),
                trace_positions=tp,
                seed=int(seed),
                **where,
            )
        except Exception as exc:  # noqa: BLE001
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
        "trace_positions": tp,
        "trace_topk": int(trace_topk),
        "config": {
            "targets": targets, "start_pos": start_pos,
            "modes": modes, "steps": steps,
            "k": int(k), "prob": probs_per_target, "seed": int(seed),
        },
    }
    # Auto-switch to the Results tab on a successful run.
    st.session_state["_pending_tab"] = "Results"
    st.rerun()

last = st.session_state.get("last_run")

# ---------------------------------------------------------------------------
# Results + Convergence tabs (rendered when their tab is active).
# ---------------------------------------------------------------------------

if active_tab == "Results":
    if last is None:
        st.info("No run yet. Fill in inputs above and click **Run experiment**.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pinned positions", len(last["positions"]))
        c2.metric("Landed text", repr(last["landed"]))
        c3.metric("All pins held?", "✅ yes" if last["all_held"] else "⚠️ no")

        st.divider()
        L, R = st.columns(2, gap="large")
        with L:
            st.markdown("#### Baseline (no steering)")
            st.write(last["baseline"])
        with R:
            st.markdown("#### Steered")
            st.write(last["steered"])

        st.divider()
        st.markdown("#### Pin survival")
        st.markdown(
            f"Steering acted on token positions **{last['positions']}**, and what "
            f"actually landed there in the final canvas was **`{last['landed']!r}`**."
        )
        st.caption(
            "If `all_held` is ✅, the attack stuck verbatim. If ⚠️, the model overrode "
            "at least one pin -- inspect the raw interventions table below to see which."
        )

        with st.expander("Raw interventions"):
            st.dataframe(pd.DataFrame(last["interventions"]), use_container_width=True)

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
            "trace": last["decoded"],
        }
        st.download_button(
            "⬇ Download run as JSON",
            data=json.dumps(payload, indent=2),
            file_name="streamlit_run.json",
            mime="application/json",
        )

if active_tab == "Convergence":
    if last is None or not last["decoded"]:
        st.info("Run an experiment with --trace enabled to see the convergence view.")
    else:
        decoded = last["decoded"]
        all_steps = sorted({rec["step_idx"] for rec in decoded})
        all_positions = sorted({int(p) for rec in decoded for p in rec["positions"]})
        steered_set = set(last["positions"])
        trace_topk_used = int(last.get("trace_topk", 8))

        st.caption(
            "DiffusionGemma denoises the whole canvas jointly over ~48 steps. Use the "
            "**step** slider to scrub through denoising; the canvas on the left shows "
            "every traced token at that step (opacity + sharpness ∝ confidence), and "
            "the right pane shows the **probability distribution** over the top candidate "
            "tokens at one focused position -- watch it narrow from spread-out to spike."
        )

        ctop1, ctop2 = st.columns([3, 2])
        step = ctop1.slider(
            "denoising step",
            min_value=int(all_steps[0]), max_value=int(all_steps[-1]),
            value=int(all_steps[-1]), step=1,
        )
        focus_pos = ctop2.selectbox(
            "focused position (drives the right-pane distribution)",
            all_positions,
            index=len(all_positions) - 1,
            format_func=lambda p: f"pos {p}" + ("  (steered)" if p in steered_set else ""),
        )

        canvas_col, dist_col = st.columns([3, 2], gap="large")

        with canvas_col:
            st.markdown(f"##### Canvas at step {step}")
            html = _step_canvas_html(decoded, step, all_positions, steered_set, focus=int(focus_pos))
            st.markdown(
                f"<div style='border:1px solid #e3e3e3;border-radius:8px;padding:18px;"
                f"background:#fff;font-family:monospace;font-size:18px;line-height:1.9;"
                f"min-height:160px'>{html}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Each glyph is one traced token position. Faint+blurred = the model is "
                "still uncertain. Bold+sharp = it has committed. "
                "<span style='color:#1f4ed8'>Blue</span> = steered (pinned). "
                "<span style='color:#b91c1c'>Red box</span> = the focused position.",
                unsafe_allow_html=True,
            )

        with dist_col:
            st.markdown(f"##### Distribution at pos {focus_pos}, step {step}")
            dist = distribution_at(decoded, step, int(focus_pos), trace_topk_used)
            pre_dist = pre_distribution_at(decoded, step, int(focus_pos), trace_topk_used)
            if dist.empty:
                st.info("No trace at this (step, position).")
            else:
                top1_prob = float(dist["prob"].iloc[0])
                entropy = -sum(p * math.log(max(p, 1e-12)) for p in dist["prob"])
                m1, m2 = st.columns(2)
                m1.metric("top-1 prob (post)", f"{top1_prob:.3f}")
                m2.metric("entropy (post)", f"{entropy:.3f}",
                          help="lower = more committed; higher = more uncertain")

                if not pre_dist.empty:
                    st.caption("**After steering** (what sampler saw)")

                def _dist_chart(df, color_scheme):
                    return (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
                            y=alt.Y("display:N", sort="-x", title=None),
                            color=alt.Color(
                                "prob:Q",
                                scale=alt.Scale(scheme=color_scheme, domain=[0, 1]),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("token:N", title="token"),
                                alt.Tooltip("prob:Q", title="prob", format=".4f"),
                            ],
                        )
                        .properties(height=max(200, 26 * len(df)))
                    )

                chart = _dist_chart(dist, "blues")
                st.altair_chart(chart, use_container_width=True)

                if not pre_dist.empty:
                    st.caption("**Before steering** (natural model distribution)")
                    pre_top1 = float(pre_dist["prob"].iloc[0])
                    pre_ent = -sum(p * math.log(max(p, 1e-12)) for p in pre_dist["prob"])
                    pm1, pm2 = st.columns(2)
                    pm1.metric("top-1 prob (pre)", f"{pre_top1:.3f}")
                    pm2.metric("entropy (pre)", f"{pre_ent:.3f}")
                    pre_chart = _dist_chart(pre_dist, "greens")
                    st.altair_chart(pre_chart, use_container_width=True)

                st.caption(
                    "Tokens shown with `·` for spaces and `⏎` for newlines so you can "
                    "see whitespace candidates. Scrub the **step** slider above and "
                    "watch the bars collapse onto the winner."
                )

        st.divider()
        st.markdown("#### Streaming-process diagnostics")
        st.caption(
            "Five complementary views of the same trace. Each one answers a different "
            "question about *how* the canvas converges, not just *what* it converges to."
        )

        d_overall, d_heat, d_commit, d_rank, d_churn, d_traj = st.tabs([
            "① Overall uncertainty",
            "② Entropy heatmap",
            "③ Commitment timeline",
            "④ Rank of final winner",
            "⑤ Top-1 churn",
            "⑥ Token trajectory table",
        ])

        with d_overall:
            st.caption(
                "**Question:** how chaotic is the canvas overall at each step? "
                "Mean entropy across all traced positions = headline uncertainty; max "
                "entropy = the worst still-undecided position. A clean run shows a "
                "monotone descent; kinks reveal moments where the model rejected a "
                "competing hypothesis. Pinned positions (entropy ≈ 0) drag mean down."
            )
            mean_df = mean_entropy_curve(decoded)
            if not mean_df.empty:
                long = mean_df.melt("step", var_name="metric", value_name="entropy")
                line = (
                    alt.Chart(long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("step:Q", title="denoising step"),
                        y=alt.Y("entropy:Q", title="entropy (nats)"),
                        color=alt.Color(
                            "metric:N",
                            scale=alt.Scale(
                                domain=["mean_entropy", "max_entropy"],
                                range=["#1f4ed8", "#b91c1c"],
                            ),
                            title=None,
                        ),
                        tooltip=["step", "metric", alt.Tooltip("entropy:Q", format=".3f")],
                    )
                    .properties(height=320)
                )
                rule = alt.Chart(pd.DataFrame({"step": [step]})).mark_rule(
                    color="#6b7280", strokeDash=[4, 3]
                ).encode(x="step:Q")
                st.altair_chart(line + rule, use_container_width=True)

            probs = top1_prob_frame(decoded)
            if not probs.empty:
                st.markdown("**Top-1 probability per traced position**")
                st.caption(
                    "One line per position. y = top-1 probability at that step. Hard "
                    "pins read as flat 1.0 from step 0; unsteered positions climb."
                )
                st.line_chart(probs, height=260)

        with d_heat:
            st.caption(
                "**Question:** which positions are unsure when? Each cell is the "
                "entropy at one (step, position). Dark = uncertain, light = decided."
            )
            ent_df = entropy_frame(decoded)
            if not ent_df.empty:
                heat = (
                    alt.Chart(ent_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("step:O", title="denoising step"),
                        y=alt.Y("position:O", title="token position", sort="ascending"),
                        color=alt.Color(
                            "entropy:Q",
                            scale=alt.Scale(scheme="magma", reverse=True),
                            title="entropy",
                        ),
                        tooltip=[
                            "step", "position",
                            alt.Tooltip("entropy:Q", format=".3f"),
                        ],
                    )
                    .properties(height=max(220, 22 * len(all_positions)))
                )
                st.altair_chart(heat, use_container_width=True)

        with d_commit:
            threshold = st.slider(
                "commitment threshold (top-1 probability)",
                min_value=0.5, max_value=0.99, value=0.9, step=0.01,
                help="A position is 'committed' once top-1 ≥ this value at some step.",
            )
            st.caption(
                "**Question:** in what order did the model lock in each position? "
                "Bar length = first step where top-1 ≥ threshold."
            )
            cf = commitment_frame(decoded, threshold=threshold)
            if not cf.empty:
                max_step = max(all_steps)
                cf2 = cf.copy()
                cf2["never"] = cf2["commit_step"].isna()
                cf2["plot_step"] = cf2["commit_step"].fillna(max_step + 1)
                cf2["pinned"] = cf2["position"].isin(steered_set)

                bars = (
                    alt.Chart(cf2)
                    .mark_bar()
                    .encode(
                        x=alt.X("plot_step:Q", title="step at which top-1 crossed threshold"),
                        y=alt.Y("position:O", sort="ascending", title="token position"),
                        color=alt.Color(
                            "never:N",
                            scale=alt.Scale(domain=[False, True], range=["#1f4ed8", "#9ca3af"]),
                            legend=alt.Legend(title=None,
                                              labelExpr="datum.value ? 'never crossed' : 'committed'"),
                        ),
                        tooltip=[
                            "position",
                            alt.Tooltip("commit_step:Q", title="commit step"),
                            alt.Tooltip("final_token:N", title="final token"),
                            alt.Tooltip("final_prob:Q", format=".3f", title="final prob"),
                            alt.Tooltip("pinned:N", title="steered?"),
                        ],
                    )
                    .properties(height=max(220, 22 * len(cf2)))
                )
                pin_dots = (
                    alt.Chart(cf2[cf2["pinned"]])
                    .mark_point(filled=True, size=80, shape="diamond", color="#f59e0b")
                    .encode(x=alt.value(2), y="position:O",
                            tooltip=[alt.Tooltip("position:O", title="pinned position")])
                )
                st.altair_chart(bars + pin_dots, use_container_width=True)
                st.caption(
                    "🔶 diamond = pinned position. Grey bar = position never reached the "
                    "threshold (rendered just past the last step for visibility)."
                )

        with d_rank:
            st.caption(
                "**Question:** when did the eventual answer first show up? At each "
                "step, we look up the rank of the *finally chosen* token at every position."
            )
            rk = final_rank_frame(decoded)
            if not rk.empty:
                rk = rk.copy()
                rk["highlight"] = rk["position"] == int(focus_pos)
                lines = (
                    alt.Chart(rk)
                    .mark_line(interpolate="step-after")
                    .encode(
                        x=alt.X("step:Q", title="denoising step"),
                        y=alt.Y("rank:Q", title="rank of final-winning token (1 = top)",
                                scale=alt.Scale(reverse=True)),
                        detail="position:N",
                        color=alt.Color(
                            "highlight:N",
                            scale=alt.Scale(domain=[True, False], range=["#b91c1c", "#cbd5e1"]),
                            legend=None,
                        ),
                        size=alt.Size(
                            "highlight:N",
                            scale=alt.Scale(domain=[True, False], range=[3, 1]),
                            legend=None,
                        ),
                        tooltip=["step", "position", "rank"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(lines, use_container_width=True)
                st.caption(
                    f"Red line = focused position ({int(focus_pos)}). "
                    "Y-axis is reversed: rank 1 (the eventual winner) is on top."
                )

        with d_churn:
            st.caption(
                "**Question:** which positions were the model most uncertain about? "
                "Bar = the count of *distinct* tokens that ever held top-1 at this position."
            )
            ch = churn_frame(decoded)
            if not ch.empty:
                ch = ch.copy()
                ch["pinned"] = ch["position"].isin(steered_set)
                bars = (
                    alt.Chart(ch)
                    .mark_bar()
                    .encode(
                        x=alt.X("distinct_top1:Q", title="# distinct tokens that held top-1"),
                        y=alt.Y("position:O", sort="ascending", title="token position"),
                        color=alt.Color(
                            "pinned:N",
                            scale=alt.Scale(domain=[True, False], range=["#1f4ed8", "#94a3b8"]),
                            legend=alt.Legend(title=None,
                                              labelExpr="datum.value === 'true' ? 'steered' : 'free'"),
                        ),
                        tooltip=[
                            "position",
                            alt.Tooltip("distinct_top1:Q", title="churn"),
                            "pinned",
                        ],
                    )
                    .properties(height=max(220, 22 * len(ch)))
                )
                st.altair_chart(bars, use_container_width=True)

        with d_traj:
            st.caption(
                "**Question:** what token was on top at each (position, step)? "
                "Heatmap-style table -- rows = positions, columns = steps."
            )
            traj = trajectory_frame(decoded)
            st.dataframe(traj, use_container_width=True)

        st.markdown("#### Film-strip (sampled steps)")
        st.caption(
            "A condensed, scrollable view of the whole denoising loop -- one row per "
            "sampled step. Opacity + blur encode the **natural** (pre-intervention) model "
            "confidence. Blue fill = a steered position; green dashed border = actively "
            "steered at that exact step. Hover any token for its natural p and, for steered "
            "steps, what the model would have picked naturally vs. what was forced."
        )
        stride = max(1, len(all_steps) // 24)
        sampled = all_steps[::stride]
        if all_steps[-1] not in sampled:
            sampled.append(all_steps[-1])
        rows_html = []
        for s in sampled:
            canvas = _step_canvas_html(decoded, s, all_positions, steered_set, focus=int(focus_pos))
            highlight = "background:#fff7d6;" if s == step else ""
            rows_html.append(
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:4px 6px;border-bottom:1px solid #eee;{highlight}"
                f"font-family:monospace;font-size:13px'>"
                f"<div style='width:64px;color:#888;font-size:11px'>step {s:>3}</div>"
                f"<div>{canvas}</div></div>"
            )
        st.markdown(
            "<div style='border:1px solid #e3e3e3;border-radius:6px;padding:4px;"
            "background:#fafafa;max-height:480px;overflow-y:auto'>"
            + "".join(rows_html)
            + "</div>",
            unsafe_allow_html=True,
        )
