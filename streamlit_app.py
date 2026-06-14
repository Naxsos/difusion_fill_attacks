"""Streamlit workbench for the diffusion fill-attack experiments.

Same `client.steer` + `example_steer.steer_strings` pipeline `run_experiments.py` and
the CLI use; the heavy ~52 GB model stays on whichever box runs `server.py`. This app
only needs `streamlit`, `pandas`, `altair`, and the lightweight tokenizer.

Layout
------
Sidebar  -> Run button + collapsed Server / Advanced settings (defaults already work).
Main     -> Three tabs:
  1. Setup      -- prompt, targets table, sampling knobs (k / prob).
  2. Results    -- baseline vs steered, pin survival, raw interventions.
  3. Convergence-- step scrubber drives a left canvas (full denoising state at the
                   selected step, opacity + blur = top-1 probability) and a right
                   "distribution sidebar" that shows the top-k probability bars at
                   the focused position so you can watch the model's belief narrow
                   from spread-out (early, unsure) to spike (late, committed).
                   Below that: film-strip overview, top-1 trajectory, top-k area.

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


# Cache the tokenizer across reruns. Streamlit reruns the script top-to-bottom on every
# interaction; without this we'd re-parse the tokenizer each click. The model is remote
# (server.py), so this only loads tokenizer files -- still worth caching.
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
    """Top-1 probability over denoising steps, one column per traced position."""
    rows: list[dict] = []
    for rec in decoded:
        row = {"step": rec["step_idx"]}
        for pos, cands in rec["positions"].items():
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

    This is the data the right-pane "distribution sidebar" plots: a snapshot of the
    sampler's belief over tokens at one moment in denoising. As `step` advances, mass
    concentrates on the winner -- rendered as a horizontal bar chart it makes the
    convergence visceral.
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


# ---------------------------------------------------------------------------
# Streaming-process diagnostics: derived frames that tell you *what* the diffusion
# is doing as it converges -- not just "what token is on top now", but how fast,
# in what order, and how chaotically.
# ---------------------------------------------------------------------------

def _entropy_of(cands: list[dict]) -> float:
    """Shannon entropy of the top-k slice. Treats the trace's top-k as the support;
    this under-estimates true entropy (we don't see the long tail) but it's monotone
    in the sharpness of the winner, which is what we care about visually.
    """
    return -sum(c["prob"] * math.log(max(c["prob"], 1e-12)) for c in cands)


def entropy_frame(decoded: list[dict]) -> pd.DataFrame:
    """Long DF (step, position, entropy) for the per-position heatmap."""
    rows = []
    for rec in decoded:
        for pos, cands in rec["positions"].items():
            if cands:
                rows.append({"step": rec["step_idx"], "position": int(pos),
                             "entropy": _entropy_of(cands)})
    return pd.DataFrame(rows)


def mean_entropy_curve(decoded: list[dict]) -> pd.DataFrame:
    """Per-step canvas-wide stats: mean and max entropy across traced positions.

    Mean is the headline -- "how unsure is the canvas overall right now". Max is the
    tail -- "is there at least one position still hedging".
    """
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
    """For each traced position, the first step where top-1 prob >= threshold.

    Positions that never crossed the threshold get `commit_step = NaN` and surface
    in the chart at the right edge so they're visible-but-flagged. Steered (pinned)
    positions tend to commit at step 0; unsteered positions reveal the *order* in
    which the model resolves the canvas.
    """
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
    """For each (step, position), the rank (1..k) of the *final winning token*.

    The "final winner" is the top-1 at the very last traced step. This frame answers
    the question "did the right answer show up early and wait, or only at the end?"
    -- a position whose final-winner-rank starts at 5 and walks down to 1 over many
    steps was actively reconsidering; one that hits rank 1 on step 0 was decided.
    Tokens not in the recorded top-k at a given step get rank = top_k + 1 (i.e.
    "off-screen") so the line stays drawable.
    """
    by_step = sorted(decoded, key=lambda r: r["step_idx"])
    if not by_step:
        return pd.DataFrame()
    last = by_step[-1]
    final_winner: dict[int, str] = {
        int(p): cands[0]["token"]
        for p, cands in last["positions"].items() if cands
    }
    rows = []
    # Determine the per-record top-k so an "off-screen" rank is one past it.
    for rec in by_step:
        for pos, cands in rec["positions"].items():
            if not cands or int(pos) not in final_winner:
                continue
            target = final_winner[int(pos)]
            rank = next((i + 1 for i, c in enumerate(cands) if c["token"] == target), len(cands) + 1)
            rows.append({"step": rec["step_idx"], "position": int(pos), "rank": rank})
    return pd.DataFrame(rows)


def churn_frame(decoded: list[dict]) -> pd.DataFrame:
    """Per position, count the number of distinct tokens that ever held top-1.

    A position with churn=1 was decided from step 0; churn=8 means the model swapped
    its #1 hypothesis seven times before settling. High churn next to a pin can
    indicate the steer is reshaping the neighborhood.
    """
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
    spans: list[str] = []
    for pos in positions:
        cands = rec["positions"].get(pos, [])
        if not cands:
            spans.append("<span style='opacity:.15;color:#999'>·</span>")
            continue
        tok = cands[0]["token"]
        prob = float(cands[0]["prob"])
        # Map probability to perceptual cues: low prob -> faint + blurred, high prob -> bold.
        opacity = 0.15 + 0.85 * prob
        blur_px = max(0.0, 3.0 * (1.0 - prob))
        weight = 700 if prob > 0.85 else 400
        if pos == focus:
            color, bg, border = "#b91c1c", "#fee2e2", "1px solid #ef4444"
        elif pos in steered_positions:
            color, bg, border = "#1f4ed8", "#eef3ff", "0"
        else:
            color, bg, border = "#111", "transparent", "0"
        display = tok.replace(" ", "·").replace("\n", "⏎") or "∅"
        spans.append(
            f"<span title='pos {pos} · p={prob:.2f}' "
            f"style='display:inline-block;margin:0 1px;padding:1px 3px;"
            f"opacity:{opacity:.3f};filter:blur({blur_px:.2f}px);"
            f"font-weight:{weight};color:{color};border:{border};border-radius:3px;"
            f"background:{bg}'>{display}</span>"
        )
    return "".join(spans)


# ---------------------------------------------------------------------------
# Targets editor: one explicit row of widgets per target, each with its own
# delete button. data_editor's row deletion is too hidden -- this surfaces it.
# ---------------------------------------------------------------------------

DEFAULT_TARGET_ROW = {"target": "Yes", "start_pos": 0, "mode": "pin", "step": 0}


def _ensure_targets_state() -> None:
    if "target_rows" not in st.session_state:
        st.session_state["target_rows"] = [dict(DEFAULT_TARGET_ROW)]
    if "targets_nonce" not in st.session_state:
        # Bumped every time we *replace* the row list wholesale (load, reset, clear);
        # baked into widget keys so old per-row widget state doesn't bleed into new rows.
        st.session_state["targets_nonce"] = 0


def _bump_nonce() -> None:
    st.session_state["targets_nonce"] = st.session_state.get("targets_nonce", 0) + 1


def _add_row_cb() -> None:
    st.session_state["target_rows"].append(dict(DEFAULT_TARGET_ROW))


def _remove_row_cb(idx: int) -> None:
    rows = st.session_state["target_rows"]
    if 0 <= idx < len(rows):
        rows.pop(idx)
    # Removing a row shifts trailing indices; bump nonce so stale widget keys are dropped.
    _bump_nonce()


def _reset_rows_cb() -> None:
    st.session_state["target_rows"] = [dict(DEFAULT_TARGET_ROW)]
    _bump_nonce()


def _clear_rows_cb() -> None:
    st.session_state["target_rows"] = []
    _bump_nonce()


def _parse_command_for_sampling(cmd: str) -> dict:
    """Pull --prob / --k / --seed out of a saved CLI command line.

    Simon's experiment files store the original `python example_steer.py ...` invocation
    in the `command` field. The structured top-level JSON keys don't always carry the
    sampling knobs, so when those are missing we fall back to scraping the command.

    Best-effort -- unknown / unparseable values are silently dropped (the loader will
    keep whatever's currently in session_state for those fields).
    """
    out: dict[str, float | int] = {}
    if not cmd:
        return out
    # Split conservatively on whitespace; quoted prompts / targets won't be parsed
    # correctly here, but we only care about the trailing flag-pairs which are simple.
    toks = cmd.split()
    for i, tok in enumerate(toks):
        if tok in ("--prob", "--probability") and i + 1 < len(toks):
            try: out["prob"] = float(toks[i + 1])
            except ValueError: pass
        elif tok in ("--k", "--top-k") and i + 1 < len(toks):
            try: out["k"] = int(toks[i + 1])
            except ValueError: pass
        elif tok == "--seed" and i + 1 < len(toks):
            try: out["seed"] = int(toks[i + 1])
            except ValueError: pass
    return out


def _load_experiment_into_state(payload: dict) -> tuple[bool, str]:
    """Pull prompt + targets + start_pos out of a Simon's-style experiment JSON
    and mirror them into session_state. Returns (ok, message).

    Schema we expect (all of Simon's files match this -- the result/verdict fields
    are optional and only used for the preview pane below):
      {
        "prompt": str,
        "targets": [str, ...],
        "start_pos": [int, ...],
        # optional, shown in the "loaded experiment" preview:
        "baseline": str, "steered": str, "landed": str, "all_held": bool,
        "verdict": int, "justification": str, "command": str, "source_txt": str,
      }
    """
    if not isinstance(payload, dict):
        return False, "Top-level JSON must be an object."
    prompt = payload.get("prompt")
    targets = payload.get("targets")
    start_pos = payload.get("start_pos")
    if not isinstance(prompt, str) or not prompt.strip():
        return False, "Missing or empty `prompt` field."
    if not isinstance(targets, list) or not targets:
        return False, "Missing or empty `targets` array."
    if not isinstance(start_pos, list) or len(start_pos) != len(targets):
        return False, "`start_pos` must be a list the same length as `targets`."

    # The Simon files don't carry per-target mode/step (they all default to pin@step 0),
    # but the workbench supports them, so we accept optional `modes` / `steps` if present.
    modes = payload.get("modes") or ["pin"] * len(targets)
    steps = payload.get("steps") or [0] * len(targets)
    if len(modes) != len(targets) or len(steps) != len(targets):
        return False, "`modes` / `steps`, if provided, must match `targets` length."

    rows = []
    for tgt, sp, m, st_ in zip(targets, start_pos, modes, steps):
        rows.append({
            "target": str(tgt),
            "start_pos": int(sp),
            "mode": m if m in ("pin", "perturb") else "pin",
            "step": int(st_),
        })

    st.session_state["target_rows"] = rows
    st.session_state["loaded_prompt"] = prompt
    # Use the joined targets as the "target output" so the user sees the goal as text.
    st.session_state["loaded_target_output"] = "".join(targets)
    # Stash the whole payload so we can show baseline / steered / verdict alongside.
    st.session_state["loaded_experiment"] = payload
    # Mine sampling knobs (k / prob / seed) out of the saved CLI command if present,
    # so loading actually reproduces the experiment instead of just its targets.
    sampling = _parse_command_for_sampling(payload.get("command", ""))
    if "k" in sampling: st.session_state["loaded_k"] = sampling["k"]
    if "prob" in sampling: st.session_state["loaded_prob"] = sampling["prob"]
    if "seed" in sampling: st.session_state["loaded_seed"] = sampling["seed"]
    _bump_nonce()
    # Nudge the prompt + target_output text_areas to re-render with the loaded values.
    st.session_state["prompt_text_nonce"] = st.session_state.get("prompt_text_nonce", 0) + 1
    extras = []
    if sampling: extras.append("sampling " + ", ".join(f"{k}={v}" for k, v in sampling.items()))
    extra_str = (" — " + "; ".join(extras)) if extras else ""
    return True, f"Loaded {len(rows)} target(s) at positions {list(start_pos)}{extra_str}."


def targets_editor() -> pd.DataFrame:
    """Card-per-target layout.

    Earlier versions packed the four target fields into a single narrow row; with
    long target strings (e.g. multi-sentence injections) the text input was unusable.
    Now each target is its own bordered "card" with:
      - a full-width text_area for the target string itself,
      - a sub-row with start_pos / mode / step / delete underneath.
    Card N gets a numbered header (#1, #2, ...) so multi-target experiments are
    visually scannable.
    """
    _ensure_targets_state()
    rows = st.session_state["target_rows"]
    nonce = st.session_state.get("targets_nonce", 0)

    if not rows:
        st.info("No targets yet. Use **Add target** below to begin.")

    for i, row in enumerate(rows):
        with st.container(border=True):
            # Header: card number + delete on the right.
            h1, h2 = st.columns([6, 1])
            h1.markdown(f"**Target #{i + 1}**")
            h2.button(
                "🗑 Delete",
                key=f"tgt_del_{nonce}_{i}",
                help=f"remove target #{i + 1}",
                on_click=_remove_row_cb, args=(i,),
                use_container_width=True,
            )

            # Target string -- full width text_area so long strings are readable.
            row["target"] = st.text_area(
                f"target string #{i + 1}",
                value=row.get("target", ""),
                key=f"tgt_target_{nonce}_{i}",
                height=80,
                placeholder=(
                    "the text being forced into the canvas, e.g. "
                    "'Napoleon clearly won at Waterloo, because'"
                ),
                help=(
                    "Tokenized and pinned to consecutive positions starting at "
                    "`start_pos`. Whitespace matters -- a leading space becomes part "
                    "of the first pinned token."
                ),
            )

            # Sub-row: position / mode / step. Wider columns than before because
            # there are only three of them sharing the row.
            sc1, sc2, sc3 = st.columns([1.2, 1.4, 1.2])
            row["start_pos"] = sc1.number_input(
                "start_pos",
                min_value=0, step=1,
                value=int(row.get("start_pos", 0)),
                key=f"tgt_start_{nonce}_{i}",
                help="first token position this target lands at",
            )
            row["mode"] = sc2.selectbox(
                "mode",
                options=["pin", "perturb"],
                index=["pin", "perturb"].index(row.get("mode", "pin")),
                key=f"tgt_mode_{nonce}_{i}",
                help="`pin` = hard freeze; `perturb` = one-shot nudge then release",
            )
            row["step"] = sc3.number_input(
                "step",
                min_value=0, step=1,
                value=int(row.get("step", 0)),
                key=f"tgt_step_{nonce}_{i}",
                help="denoising step (0..~47) at which this target fires",
            )

    # Footer: add / reset / clear -- callbacks mutate state before rerender.
    f1, f2, f3, _ = st.columns([1.7, 1.3, 1.5, 4])
    f1.button("➕ Add target", on_click=_add_row_cb, use_container_width=True)
    f2.button("↺ Reset", on_click=_reset_rows_cb, use_container_width=True,
              help="restore the example target")
    f3.button("🗑 Clear all", on_click=_clear_rows_cb, use_container_width=True,
              help="remove every target")

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["target", "start_pos", "mode", "step"]
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DiffusionGemma fill-attack workbench", layout="wide")
st.title("DiffusionGemma fill-attack workbench")

# --- Sidebar: run button + collapsed settings ------------------------------
with st.sidebar:
    st.markdown("### ▶ Run")
    run = st.button("Run experiment", type="primary", use_container_width=True)
    if "last_run" in st.session_state:
        st.caption(f"Last run: **{len(st.session_state['last_run'].get('decoded', []))}** trace records")

    with st.expander("Server", expanded=False):
        st.caption(
            "ℹ️ Defaults match the bundled `server.py`. "
            "**No change needed** unless your server is on a different host/port."
        )
        host = st.text_input("host", value="localhost")
        port = st.number_input("port", value=8000, step=1)

    with st.expander("Advanced", expanded=False):
        st.caption("Reproducibility + tracing knobs. Defaults are usually fine.")
        seed = st.number_input(
            "seed",
            value=int(st.session_state.pop("loaded_seed", st.session_state.get("seed_val", 0))),
            step=1,
            key=f"seed_input_{st.session_state.get('prompt_text_nonce', 0)}",
        )
        st.session_state["seed_val"] = seed
        trace_topk = st.number_input(
            "trace topk", min_value=1, value=8, step=1,
            help="how many candidates to record per traced position per step",
        )
        extra_positions = st.text_input(
            "extra trace positions", value="",
            help="comma-separated extra positions to track besides the steered ones",
        )

# --- Tabs -------------------------------------------------------------------
tab_setup, tab_results, tab_converge = st.tabs(
    ["⚙️ Setup", "📊 Results", "🌫️ Convergence"]
)

# --- SETUP TAB --------------------------------------------------------------
with tab_setup:
    # File uploader sits *above* the two-column setup so it's the first thing you
    # see -- "load an experiment and we fill the form for you" is a faster path than
    # typing prompts/targets by hand. The Simon's experiments in simons_experiments/
    # all match the schema we accept.
    with st.expander("📂 Load experiment from JSON", expanded=False):
        st.caption(
            "Upload a Simon's-style experiment JSON (e.g. files in `simons_experiments/`). "
            "The **prompt**, **target output**, and **Targets** rows below will be filled "
            "from the file. If the file also contains saved `baseline` / `steered` text "
            "and a judge `verdict`, you'll see them in a preview pane after loading."
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
                    st.success(f"✅ {uploaded.name} -- {msg}")
                    # Bump the uploader nonce so the same file can be re-uploaded later;
                    # otherwise Streamlit keeps the file handle and won't trigger again.
                    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
                    st.rerun()
                else:
                    st.error(f"⚠ {uploaded.name}: {msg}")
            except json.JSONDecodeError as e:
                st.error(f"Could not parse JSON: {e}")

        # If we have a previously-loaded experiment, show its saved results so the user
        # can compare without re-running the model.
        loaded = st.session_state.get("loaded_experiment")
        if loaded:
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Saved baseline (from file)**")
                st.caption("The output the file recorded for this prompt without steering.")
                st.write(loaded.get("baseline", "_(not in file)_"))
            with colB:
                st.markdown("**Saved steered (from file)**")
                st.caption("The output recorded after the steer was applied.")
                st.write(loaded.get("steered", "_(not in file)_"))
            verdict = loaded.get("verdict")
            justification = loaded.get("justification")
            if verdict is not None or justification:
                m1, m2, m3 = st.columns(3)
                if verdict is not None:
                    m1.metric("judge verdict", verdict)
                if loaded.get("integration") is not None:
                    m2.metric("integration", loaded["integration"])
                if loaded.get("stance_adopted") is not None:
                    m3.metric("stance adopted", loaded["stance_adopted"])
                if justification:
                    with st.expander("Judge justification"):
                        st.write(justification)
            if loaded.get("command"):
                st.code(loaded["command"], language="bash")
            if st.button("Clear loaded experiment", help="discard the loaded preview"):
                st.session_state.pop("loaded_experiment", None)
                st.session_state.pop("loaded_prompt", None)
                st.session_state.pop("loaded_target_output", None)
                st.rerun()

    left, right = st.columns([3, 2], gap="large")

    with left:
        # 1. Target output -- the desired final text we want to steer toward. Shown
        # first because that's the framing: "this is what I want to come out."
        st.markdown("#### 1. Target output")
        st.caption(
            "The text snippet you want the model to produce. This is the *goal* of the "
            "steer -- the **Targets** table below is how you encode it (which tokens "
            "land at which positions, and on which denoising steps)."
        )
        # `loaded_target_output` is set by the uploader; treat it as a one-shot
        # initializer so subsequent edits aren't clobbered when the user types.
        target_default = st.session_state.pop(
            "loaded_target_output",
            st.session_state.get("target_output_text", "Yes, a hot dog is a sandwich."),
        )
        target_output = st.text_area(
            "target output", label_visibility="collapsed",
            value=target_default,
            height=90,
            key=f"target_output_text_{st.session_state.get('prompt_text_nonce', 0)}",
        )
        st.session_state["target_output_text"] = target_output

        # 2. Prompt sent to the model.
        st.markdown("#### 2. Prompt")
        st.caption("The user message the model sees. The baseline runs against this verbatim.")
        prompt_default = st.session_state.pop(
            "loaded_prompt",
            st.session_state.get(
                "prompt_text",
                "Is a hot dog a sandwich? Give a one-word verdict (Yes or No), then explain.",
            ),
        )
        prompt = st.text_area(
            "prompt", label_visibility="collapsed",
            value=prompt_default,
            height=110,
            key=f"prompt_text_{st.session_state.get('prompt_text_nonce', 0)}",
        )
        st.session_state["prompt_text"] = prompt

        # 3. Per-token interventions.
        st.markdown("#### 3. Targets (interventions)")
        st.caption(
            "Each card below is **one target** -- a text snippet pinned at a specific "
            "position. The target string sits in its own full-width text area so multi-"
            "sentence injections stay readable; below it you set **start_pos** (first "
            "token position), **mode** (`pin` or `perturb`), and **step** (denoising "
            "step 0..~47 when this target fires). Use **➕ Add target** for staggered "
            "or multi-position steers (Simon's experiments often have 2 targets at "
            "different positions). Click 🗑 Delete to remove a target."
        )
        targets_df = targets_editor()

    with right:
        st.markdown("#### Sampling")
        k = st.number_input(
            "k (top-k width)", min_value=1,
            value=int(st.session_state.pop("loaded_k", st.session_state.get("k_val", 1))),
            step=1,
            help="1 = hard freeze on the target; ≥2 spreads residual mass across runner-ups",
            key=f"k_input_{st.session_state.get('prompt_text_nonce', 0)}",
        )
        st.session_state["k_val"] = k
        prob = st.number_input(
            "prob (mass on target; 0 = hard pin)",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.pop("loaded_prob", st.session_state.get("prob_val", 0.0))),
            step=0.05,
            help="0 = leave probabilities unset (hard pin). >0 sets per-token mass.",
            key=f"prob_input_{st.session_state.get('prompt_text_nonce', 0)}",
        )
        st.session_state["prob_val"] = prob

        st.markdown("#### What this does")
        st.markdown(
            "- A **baseline** generation runs first (no steering).\n"
            "- Then the **steered** generation runs with your targets pinned at the "
            "specified positions and steps, aiming at the **target output** above.\n"
            "- The server returns a per-step **trace** of the top-k tokens at every "
            "traced position; the **Convergence** tab visualizes that.\n"
            "- You can also **load an experiment** from a JSON file at the top of "
            "this tab to populate prompt + targets in one click."
        )
        st.info("Click **Run experiment** in the sidebar →", icon="▶")

# --- RUN --------------------------------------------------------------------
if run:
    df = targets_df.dropna(subset=["target"]).copy()
    df = df[df["target"].astype(str).str.len() > 0]
    if df.empty:
        st.error("Add at least one target row in the Setup tab.")
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
        except Exception as exc:  # noqa: BLE001
            st.error(f"Server call failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()

    decoded = decode_trace(result.get("trace", []), tokenizer) if result.get("trace") else []
    landed = "".join(o["actual_token"] for o in result["interventions"])
    st.session_state["last_run"] = {
        "target_output": target_output,
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
    st.toast("Run complete -- see the Results / Convergence tabs.", icon="✅")

last = st.session_state.get("last_run")

# --- RESULTS TAB ------------------------------------------------------------
with tab_results:
    if last is None:
        st.info("No run yet. Configure on **Setup**, then click **Run experiment** in the sidebar.")
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

        # Export tucked into Results since it's a result artifact.
        st.divider()
        decoded_for_export = last["decoded"]
        payload = {
            "target_output": last.get("target_output", ""),
            "prompt": last["prompt"],
            "config": last["config"],
            "baseline": last["baseline"],
            "steered": last["steered"],
            "landed": last["landed"],
            "positions": last["positions"],
            "all_held": last["all_held"],
            "interventions": last["interventions"],
            "trace_positions": last["trace_positions"],
            "trace": decoded_for_export,
        }
        st.download_button(
            "⬇ Download run as JSON",
            data=json.dumps(payload, indent=2),
            file_name="streamlit_run.json",
            mime="application/json",
        )

# --- CONVERGENCE TAB --------------------------------------------------------
with tab_converge:
    if last is None or not last["decoded"]:
        st.info("Run an experiment with tracing enabled to see the convergence view.")
    else:
        decoded = last["decoded"]
        all_steps = sorted({rec["step_idx"] for rec in decoded})
        all_positions = sorted({int(p) for rec in decoded for p in rec["positions"]})
        steered_set = set(last["positions"])

        st.caption(
            "DiffusionGemma denoises the whole canvas jointly over ~48 steps. Use the "
            "**step** slider to scrub through denoising; the canvas on the left shows "
            "every traced token at that step (opacity + sharpness ∝ confidence), and "
            "the right pane shows the **probability distribution** over the top candidate "
            "tokens at one focused position -- watch it narrow from spread-out to spike."
        )

        # --- Top controls: step + position pickers, side by side ----------
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

        # --- Two-pane main view: canvas | distribution --------------------
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
            dist = distribution_at(decoded, step, int(focus_pos), int(trace_topk))
            if dist.empty:
                st.info("No trace at this (step, position).")
            else:
                # Horizontal bar chart -- bars sorted by probability so the winner is on
                # top. Bar length is the actual probability (0..1) so cross-step comparison
                # is honest: an "uncertain" step has all bars short; a "committed" step has
                # one bar near 1.0 and the rest near 0.
                top1_prob = float(dist["prob"].iloc[0])
                # Shannon entropy over the top-k slice. Lower = more committed (a sharp
                # winner), higher = more spread out (the model is still hedging).
                entropy = -sum(p * math.log(max(p, 1e-12)) for p in dist["prob"])
                m1, m2 = st.columns(2)
                m1.metric("top-1 prob", f"{top1_prob:.3f}")
                m2.metric("entropy", f"{entropy:.3f}",
                          help="lower = more committed; higher = more uncertain")

                chart = (
                    alt.Chart(dist)
                    .mark_bar()
                    .encode(
                        x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
                        y=alt.Y("display:N", sort="-x", title=None),
                        color=alt.Color(
                            "prob:Q",
                            scale=alt.Scale(scheme="blues", domain=[0, 1]),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("token:N", title="token"),
                            alt.Tooltip("prob:Q", title="prob", format=".4f"),
                        ],
                    )
                    .properties(height=max(200, 26 * len(dist)))
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    "Tokens shown with `·` for spaces and `⏎` for newlines so you can "
                    "see whitespace candidates. Scrub the **step** slider above and "
                    "watch the bars collapse onto the winner."
                )

        # --- Below the fold: streaming-process diagnostics ---------------
        # Each chart answers a different question about the denoising. Tabs keep them
        # navigable without making the page a giant scroll.
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

        # ① One-line headline: how unsure is the canvas overall, step by step?
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
                # Vertical rule at the currently-selected scrubber step so this chart
                # stays in sync with the upstream canvas/distribution view.
                rule = alt.Chart(pd.DataFrame({"step": [step]})).mark_rule(
                    color="#6b7280", strokeDash=[4, 3]
                ).encode(x="step:Q")
                st.altair_chart(line + rule, use_container_width=True)

            # Bonus: per-position top-1 confidence lines, exact data, slightly different
            # framing -- "is this position done yet?" Useful next to the entropy curve.
            probs = top1_prob_frame(decoded)
            if not probs.empty:
                st.markdown("**Top-1 probability per traced position**")
                st.caption(
                    "One line per position. y = top-1 probability at that step. Hard "
                    "pins read as flat 1.0 from step 0; unsteered positions climb."
                )
                st.line_chart(probs, height=260)

        # ② 2D entropy: step × position. The "wave of commitment" view.
        with d_heat:
            st.caption(
                "**Question:** which positions are unsure when? Each cell is the "
                "entropy at one (step, position). Dark = uncertain, light = decided. "
                "You see commitment ripple across positions, and you can spot regions "
                "of the canvas that stayed contested longer than others."
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

        # ③ For each position, the first step where it crossed a confidence threshold.
        with d_commit:
            threshold = st.slider(
                "commitment threshold (top-1 probability)",
                min_value=0.5, max_value=0.99, value=0.9, step=0.01,
                help="A position is 'committed' once top-1 ≥ this value at some step.",
            )
            st.caption(
                "**Question:** in what order did the model lock in each position? "
                "Each bar is one position; bar length = first step where top-1 ≥ "
                "threshold. Positions that never cross the threshold are flagged. "
                "Steered (pinned) positions usually commit at step 0; the spread of "
                "the rest reveals how the diffusion resolves the canvas left-to-right, "
                "in clusters, or randomly."
            )
            cf = commitment_frame(decoded, threshold=threshold)
            if not cf.empty:
                # Bin: committed (numeric step) vs never-committed (NaN -> render at max+1
                # with a different color so they stay visible).
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
                # Mark pinned positions with a small dot so you can spot them on the chart.
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

        # ④ For each position, the rank of the final-winning token over time.
        with d_rank:
            st.caption(
                "**Question:** when did the eventual answer first show up? At each "
                "step, we look up the rank of the *finally chosen* token at every "
                "position. A line that starts at rank 1 means the model knew from the "
                "start; a line that descends from rank 5→4→...→1 means it actively "
                "swapped its mind. Lines pinned to rank 1 from step 0 are likely steered."
            )
            rk = final_rank_frame(decoded)
            if not rk.empty:
                # Highlight the focused position; others go light grey so the chart is
                # legible even with many lines.
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
                    f"Red line = focused position ({int(focus_pos)}). Lines ride near "
                    "the top (rank 1) only after the model has decided this position. "
                    "Y-axis is reversed: rank 1 (the eventual winner) is on top."
                )

        # ⑤ How many distinct tokens ever won top-1, per position.
        with d_churn:
            st.caption(
                "**Question:** which positions were the model most uncertain about? "
                "Bar = the count of *distinct* tokens that ever held top-1 at this "
                "position across the whole run. 1 = decided from step 0 and never "
                "moved. 5+ = the model thrashed -- often interesting to look at "
                "neighbors of pinned positions, where the steer is reshaping context."
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

        # ⑥ The original token-trajectory table -- still the ground-truth lookup.
        with d_traj:
            st.caption(
                "**Question:** what token was on top at each (position, step)? "
                "Heatmap-style table -- rows = positions, columns = steps. Use this "
                "as a ground-truth lookup when one of the charts above raises a question."
            )
            traj = trajectory_frame(decoded)
            st.dataframe(traj, use_container_width=True)

        st.markdown("#### Film-strip (sampled steps)")
        st.caption(
            "A condensed, scrollable view of the whole denoising loop -- one row per "
            "sampled step. Same opacity + blur encoding as the main canvas."
        )
        # Stride to keep ~24 rows max; always include the last step.
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
