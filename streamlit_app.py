"""Streamlit workbench for the diffusion fill-attack experiments — simple form UI.

The input UI is a flat form: one widget per CLI flag emitted by
``steer_config.to_cli``. The Results and Convergence views are unchanged from
the previous workbench.

Launch (the diffusion server must already be running -- see SERVER.md):

    pip install streamlit pandas altair
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import datetime
import glob
import html
import json
import math
import os
import traceback

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


def topk_at_position_frame(
    decoded: list[dict], position: int, top_k: int, prefer_pre: bool = False
) -> pd.DataFrame:
    """Top-k token probabilities at one position over EVERY denoising step (wide format:
    index = step, one column per candidate token).

    With ``prefer_pre`` it reports the natural (pre-intervention) distribution where the
    recorder captured one, falling back to the post-intervention distribution otherwise --
    so a steered position shows the model's genuine uncertainty instead of the forced spike.
    """
    rows: list[dict] = []
    for rec in decoded:
        src = (rec.get("pre_positions") if prefer_pre else None) or rec["positions"]
        cands = src.get(position, [])
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


def enrich_interventions(
    interventions: list[dict], decoded: list[dict]
) -> pd.DataFrame:
    """Join per-step trace confidences onto the interventions table.

    ``p`` and ``k`` on the raw interventions are the **inputs** to the steer (the
    requested ``--prob`` and top-k width), so they look constant across rows. The
    interesting "how confident was the model?" signal lives in the trace. For each
    intervention we look up, at its position:

    - ``nat_top1_token_final`` / ``nat_top1_prob_final`` -- what the unsteered model
      *wanted* to put there at the last denoising step, and how confident it was
    - ``nat_prob_of_requested_final`` -- natural prob the model assigned to the token
      we forced (i.e. how plausible was our pin to the model on its own?)
    - ``nat_top1_prob_max`` / ``nat_top1_prob_first`` -- range of natural confidence
      across the whole denoising trajectory at this position (first step vs peak)
    - ``post_top1_prob_final`` -- post-intervention top-1 prob at the last step
      (≈1.0 for held hard pins)
    """
    df = pd.DataFrame(interventions)
    if df.empty or not decoded:
        return df

    by_step = sorted(decoded, key=lambda r: r["step_idx"])
    last = by_step[-1]
    first = by_step[0]

    def _lookup(pos: int, requested_id: int) -> dict:
        out = {
            "nat_top1_token_final": None,
            "nat_top1_prob_final": None,
            "nat_prob_of_requested_final": None,
            "nat_top1_prob_first": None,
            "nat_top1_prob_max": None,
            "post_top1_prob_final": None,
        }
        # Natural distribution at the last step.
        pre_last = (last.get("pre_positions") or {}).get(pos) or []
        if pre_last:
            out["nat_top1_token_final"] = pre_last[0]["token"]
            out["nat_top1_prob_final"] = float(pre_last[0]["prob"])
            req = next((c for c in pre_last if int(c["id"]) == int(requested_id)), None)
            if req is not None:
                out["nat_prob_of_requested_final"] = float(req["prob"])

        # Natural top-1 prob at the first step (what the model thought before any
        # denoising progress was made at this position).
        pre_first = (first.get("pre_positions") or {}).get(pos) or []
        if pre_first:
            out["nat_top1_prob_first"] = float(pre_first[0]["prob"])

        # Peak natural top-1 prob across the trajectory.
        peak = 0.0
        seen_any = False
        for rec in by_step:
            pre_pos = (rec.get("pre_positions") or {}).get(pos) or []
            if pre_pos:
                seen_any = True
                peak = max(peak, float(pre_pos[0]["prob"]))
        if seen_any:
            out["nat_top1_prob_max"] = peak

        # Post-intervention top-1 prob at the last step (sanity for "did the pin hold loud?").
        post_last = (last.get("positions") or {}).get(pos) or []
        if post_last:
            out["post_top1_prob_final"] = float(post_last[0]["prob"])

        return out

    enriched_rows = []
    for _, row in df.iterrows():
        extra = _lookup(int(row["position"]), int(row["requested_id"]))
        enriched_rows.append({**row.to_dict(), **extra})
    return pd.DataFrame(enriched_rows)


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
# Step canvas: render every traced position at one step. The most-likely token
# sits on a GREEN background whose intensity ∝ its probability (super green at
# p=1, pale at p≈0), so "uncertain early, certain late" reads at a glance.
# Injected / pinned positions are coloured BLUE instead of green.
# ---------------------------------------------------------------------------

def _mix(base: tuple[int, int, int], target: tuple[int, int, int], t: float) -> str:
    """Linear blend from `base` to `target` by `t` in [0,1], as a CSS rgb() string."""
    t = max(0.0, min(1.0, t))
    r, g, b = (round(c0 + (c1 - c0) * t) for c0, c1 in zip(base, target))
    return f"rgb({r},{g},{b})"


def _mix_themed(target: tuple[int, int, int], t: float) -> str:
    """Like `_mix` but returns a CSS `light-dark(...)` value: blends from white in
    light mode and from a dark surface in dark mode, so the canvas reads on both
    themes without re-rendering. The page must opt in via `color-scheme: light dark`.
    """
    return f"light-dark({_mix(_CANVAS_WHITE, target, t)}, {_mix(_CANVAS_DARK, target, t)})"


_CANVAS_WHITE = (255, 255, 255)
_CANVAS_DARK = (30, 30, 33)     # dark-mode surface (Streamlit dark base ≈ #1e1e21)
_CANVAS_GREEN = (22, 163, 74)   # super green at p=1 (confident, natural)
_CANVAS_BLUE = (37, 99, 235)    # injected / pinned position


def _render_canvas_iframe(
    inner_html: str, *, height: int, frame_style: str, scrolling: bool = False,
) -> None:
    """Render canvas HTML inside a components iframe with a click handler.

    Token clicks navigate the *parent* page to `?focus=N&step=M`. That triggers a
    full Streamlit page reload — but the most recent run is auto-restored from
    `frontend_runs/*.json` on page-load (see `_autoload_last_run`), so to the user
    it just looks like the focus dropdown moved. This is the most robust approach
    given Streamlit's iframe sandboxing: programmatic URL changes via
    `pushState`+`popstate` are not reliably picked up by Streamlit's frontend, but
    a real navigation always is.

    With ``scrolling=True`` the iframe itself becomes scrollable so canvases
    taller than ``height`` stay reachable instead of being clipped.
    """
    # When the caller wants scrolling, let the iframe body scroll vertically;
    # otherwise we keep the historical fixed-height behavior.
    body_overflow = "auto" if scrolling else "hidden"
    doc = f"""
<!doctype html>
<html><head><style>
  :root{{
    color-scheme:light dark;
    --df-fg:#111827; --df-muted:#6b7280; --df-border:rgba(0,0,0,0.08);
  }}
  @media (prefers-color-scheme: dark) {{
    :root{{
      --df-fg:#e5e7eb; --df-muted:#9ca3af; --df-border:rgba(255,255,255,0.12);
    }}
  }}
  html,body{{margin:0;padding:0;background:transparent;color:var(--df-fg);
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    overflow-y:{body_overflow};}}
  .df-frame{{{frame_style}}}
  .df-tok{{transition:outline 80ms;}}
  .df-tok:hover{{outline:2px solid #6366f1; outline-offset:1px;}}
</style></head>
<body>
  <div class="df-frame">{inner_html}</div>
  <script>
    document.addEventListener('click', (ev) => {{
      const el = ev.target.closest('.df-tok');
      if (!el) return;
      const pos = el.getAttribute('data-pos');
      if (pos === null) return;
      // Navigate the parent: rewrites the URL with ?focus=N (preserving the
      // current step query param if present). Streamlit reruns; the Python side
      // reads st.query_params and the auto-restore brings `last_run` back from
      // disk so the UI just shows the new focus.
      const url = new URL(window.parent.location.href);
      url.searchParams.set('focus', String(parseInt(pos, 10)));
      window.parent.location.href = url.toString();
    }}, true);
  </script>
</body></html>
"""
    components.html(doc, height=height, scrolling=scrolling)


def _autoload_last_run() -> bool:
    """If session_state has no `last_run`, try to restore the most recent
    `frontend_runs/*.json`. Returns True iff a run was restored."""
    if "last_run" in st.session_state:
        return False
    runs_dir = os.path.join(os.path.dirname(__file__), "frontend_runs")
    if not os.path.isdir(runs_dir):
        return False
    files = sorted(
        glob.glob(os.path.join(runs_dir, "*.json")),
        key=os.path.getmtime, reverse=True,
    )
    if not files:
        return False
    try:
        with open(files[0]) as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    ok, _ = _load_run_for_viz(payload)
    return ok


def _step_canvas_html(decoded: list[dict], step_idx: int, positions: list[int],
                      steered_positions: set[int], focus: int | None = None) -> str:
    rec = next((r for r in decoded if r["step_idx"] == step_idx), None)
    if rec is None:
        return "<em>(no record at this step)</em>"

    # Positions actively steered *at this specific step* (e.g. a perturb firing now).
    active_steered: set[int] = set(rec.get("steered_positions", []))
    pre_pos = rec.get("pre_positions")  # natural distribution before intervention

    spans: list[str] = []
    for pos in positions:
        cands = rec["positions"].get(pos, [])
        # Each cell carries `data-pos` so the iframe-level click listener (installed by
        # `_canvas_click_bridge`) can read which position was clicked and forward it to
        # the parent page as a query-param + popstate, which Streamlit reruns on.
        if not cands:
            spans.append(
                f"<span class='df-tok' data-pos='{pos}' "
                f"style='cursor:pointer;opacity:.4;color:var(--df-muted)'>·</span>"
            )
            continue

        tok = cands[0]["token"]
        post_prob = float(cands[0]["prob"])  # likelihood of the shown (most-likely) token
        display = tok.replace(" ", "·").replace("\n", "⏎") or "∅"

        # Blue for injected/pinned positions, green-by-probability otherwise. A pinned
        # position keeps a clearly-blue floor so it stays readable even when its prob dips.
        is_pinned = pos in steered_positions or pos in active_steered
        if is_pinned:
            t = max(post_prob, 0.45)
            bg = _mix_themed(_CANVAS_BLUE, t)
        else:
            t = post_prob
            bg = _mix_themed(_CANVAS_GREEN, t)
        # Above ~0.55 mix, the colored fill is dark enough that white text reads well in
        # both themes; below that we let the surface text color (light or dark mode aware)
        # take over via `--df-fg`.
        txt = "#fff" if t > 0.55 else "var(--df-fg)"
        weight = 700 if post_prob > 0.85 else 500

        # Border: red = the focused position; dashed blue = steered *this very step*.
        if pos == focus:
            border = "2px solid #ef4444"
        elif pos in active_steered:
            border = "2px dashed #60a5fa"
        else:
            border = "1px solid var(--df-border)"

        # Tooltip: post prob, plus the natural (pre-intervention) token/prob when steered.
        if pre_pos is not None and is_pinned:
            pre_cands = pre_pos.get(pos, [])
            nat_tok = pre_cands[0]["token"] if pre_cands else "?"
            nat_p = float(pre_cands[0]["prob"]) if pre_cands else 0.0
            title = (
                f"pos {pos} | natural: '{nat_tok.replace(chr(39), '`')}' p={nat_p:.2f}"
                f" → steered: p={post_prob:.2f}"
            )
        else:
            title = f"pos {pos} · p={post_prob:.2f}"

        spans.append(
            f"<span class='df-tok' data-pos='{pos}' title='{title}' "
            f"style='display:inline-block;margin:1px;padding:1px 4px;cursor:pointer;"
            f"font-weight:{weight};color:{txt};border:{border};border-radius:3px;"
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
    if "rows_nonce" not in st.session_state:
        st.session_state["rows_nonce"] = 0
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []


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

st.set_page_config(
    page_title="DiffusionGemma fill-attack workbench",
    page_icon="🧪",
    layout="wide",
)

# Tighten the default Streamlit chrome and give the workbench a quieter, more consistent
# visual rhythm: tighter heading margins, denser metric cards, divider lines that don't
# scream, and a clearer "active tab" pill.
st.markdown(
    """
    <style>
      /* Opt the page in to native CSS theming so `light-dark()` works in inline
         styles (canvas spans, header card, metric tiles). Streamlit follows the
         OS / user-chosen theme; this just lets our CSS see which one is active. */
      :root {
        color-scheme: light dark;
        --df-fg: #111827;
        --df-muted: #6b7280;
        --df-faint: #9ca3af;
        --df-border: rgba(0,0,0,0.08);
        --df-card-bg: #fafafa;
        --df-card-border: #eee;
        --df-divider: #ececec;
        --df-h4: #555;
        --df-tab-active-bg: #eef2ff;
        --df-tab-active-fg: #1e293b;
        --df-tab-active-border: #c7d2fe;
        --df-tab-active-bg-hover: #e0e7ff;
        --df-tab-active-border-hover: #a5b4fc;
        --df-tab-secondary-fg: #6b7280;
        --df-tab-secondary-border: #e5e7eb;
        --df-tab-secondary-bg-hover: #f9fafb;
        --df-tab-secondary-fg-hover: #1f2937;
        --df-caption: #777;
        --df-canvas-frame: #e3e3e3;
        --df-canvas-bg: #fff;
        --df-strip-bg: #fafafa;
        --df-strip-row-border: #eee;
        --df-strip-highlight: #fff7d6;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --df-fg: #e5e7eb;
          --df-muted: #9ca3af;
          --df-faint: #6b7280;
          --df-border: rgba(255,255,255,0.12);
          --df-card-bg: #1f2024;
          --df-card-border: #2a2c31;
          --df-divider: #2a2c31;
          --df-h4: #cbd5e1;
          --df-tab-active-bg: #312e81;
          --df-tab-active-fg: #e0e7ff;
          --df-tab-active-border: #4338ca;
          --df-tab-active-bg-hover: #3730a3;
          --df-tab-active-border-hover: #6366f1;
          --df-tab-secondary-fg: #9ca3af;
          --df-tab-secondary-border: #2a2c31;
          --df-tab-secondary-bg-hover: #1f2024;
          --df-tab-secondary-fg-hover: #e5e7eb;
          --df-caption: #9ca3af;
          --df-canvas-frame: #2a2c31;
          --df-canvas-bg: #18191c;
          --df-strip-bg: #18191c;
          --df-strip-row-border: #2a2c31;
          --df-strip-highlight: #3a3318;
        }
      }

      .block-container {padding-top: 2rem !important; padding-bottom: 2rem !important;
                        max-width: 1400px;}
      header[data-testid="stHeader"] {height: 3rem;}
      h1 {font-size: 1.55rem !important; font-weight: 600 !important;
          margin: 0.1rem 0 0.1rem 0 !important;}
      h2, h3 {margin-top: 0.5rem !important; margin-bottom: 0.3rem !important;}
      h4 {margin-top: 0.3rem !important; margin-bottom: 0.2rem !important;
          font-weight: 600 !important; color: var(--df-h4);}
      hr {margin: 0.6rem 0 !important; border-color: var(--df-card-border) !important;}
      div[data-testid="stTabs"] {margin-top: 0.2rem;}
      /* Metric cards: a subtle border so they read as cards rather than floating numbers. */
      div[data-testid="stMetric"] {
        background: var(--df-card-bg); border: 1px solid var(--df-card-border);
        border-radius: 8px; padding: 0.5rem 0.8rem;
      }
      div[data-testid="stMetricLabel"] {font-size: 0.78rem !important;
        color: var(--df-muted) !important;}
      /* Top-tab buttons: pill-shaped, with the active one a saturated primary. */
      div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: transparent; border: 1px solid var(--df-tab-secondary-border);
      }
      /* Code blocks (CLI preview): a touch denser. */
      pre {padding: 0.6rem !important; font-size: 0.82rem !important;}
      /* Captions: a hair smaller, calmer color. */
      div[data-testid="stCaption"] {color: var(--df-caption) !important;
        font-size: 0.78rem !important;}

      /* ---- Quieter primary-button palette across the dashboard. The Streamlit default
         is a saturated red that screams "destructive" -- on a research workbench the
         primary action is just "go", not "delete". A calm indigo/slate reads as
         professional without losing emphasis. Applied to ALL primary buttons (tabs,
         "Run experiment", "New run", "Clear & import"); inactive tabs use a near-
         transparent secondary look. */
      button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: #4f46e5 !important; border-color: #4338ca !important;
        color: #ffffff !important; box-shadow: none !important;
      }
      button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background: #4338ca !important; border-color: #3730a3 !important;
      }
      /* Tabs in particular: even calmer, since the active tab is a selection state, not
         a call-to-action. We target the buttons by their explicit `key=` (Streamlit
         adds `st-key-<key>` as a class to each keyed widget's wrapper div), so this
         reliably scopes ONLY to the tab buttons -- "Run experiment" and "New run"
         stay on the indigo primary palette above. */
      div.st-key-tab_btn_Setup button[kind="primary"],
      div.st-key-tab_btn_Results button[kind="primary"],
      div.st-key-tab_btn_Convergence button[kind="primary"],
      div.st-key-tab_btn_Setup button[data-testid="stBaseButton-primary"],
      div.st-key-tab_btn_Results button[data-testid="stBaseButton-primary"],
      div.st-key-tab_btn_Convergence button[data-testid="stBaseButton-primary"] {
        background: var(--df-tab-active-bg) !important;
        color: var(--df-tab-active-fg) !important;
        border: 1px solid var(--df-tab-active-border) !important;
        font-weight: 600 !important; box-shadow: none !important;
      }
      div.st-key-tab_btn_Setup button[kind="primary"]:hover,
      div.st-key-tab_btn_Results button[kind="primary"]:hover,
      div.st-key-tab_btn_Convergence button[kind="primary"]:hover {
        background: var(--df-tab-active-bg-hover) !important;
        border-color: var(--df-tab-active-border-hover) !important;
        color: var(--df-tab-active-fg) !important;
      }
      div.st-key-tab_btn_Setup button[kind="secondary"],
      div.st-key-tab_btn_Results button[kind="secondary"],
      div.st-key-tab_btn_Convergence button[kind="secondary"] {
        background: transparent !important;
        color: var(--df-tab-secondary-fg) !important;
        border: 1px solid var(--df-tab-secondary-border) !important;
        box-shadow: none !important;
      }
      div.st-key-tab_btn_Setup button[kind="secondary"]:hover,
      div.st-key-tab_btn_Results button[kind="secondary"]:hover,
      div.st-key-tab_btn_Convergence button[kind="secondary"]:hover {
        background: var(--df-tab-secondary-bg-hover) !important;
        color: var(--df-tab-secondary-fg-hover) !important;
      }

    </style>
    """,
    unsafe_allow_html=True,
)

# Page header. Two-line title block: product name + a one-line subtitle, plus a quiet
# right-aligned model badge -- gives the dashboard a "product" feel rather than a script.
st.markdown(
    """
    <div style="display:flex; align-items:flex-start; justify-content:space-between;
                padding: 0.4rem 0 0.6rem 0; border-bottom: 1px solid var(--df-divider);
                margin-bottom: 0.9rem;">
      <div>
        <div style="display:flex; align-items:center; gap:0.55rem;">
          <span style="font-size:1.4rem;">🧪</span>
          <span style="font-size:0.78rem; letter-spacing:0.16em; color:var(--df-muted);
                       text-transform:uppercase; font-weight:600;">
            Diffusion Steering Lab
          </span>
        </div>
        <div style="font-size:1.45rem; font-weight:600; color:var(--df-fg);
                    margin-top:0.15rem; line-height:1.25;">
          Fill-Attack Workbench
        </div>
        <div style="color:var(--df-muted); font-size:0.85rem; margin-top:0.15rem; max-width:60ch;">
          Pin tokens at fixed canvas positions during denoising and watch the model
          rationalize around them.
        </div>
      </div>
      <div style="text-align:right; color:var(--df-faint); font-size:0.75rem;
                  padding-top:0.4rem; line-height:1.4;">
        <div style="text-transform:uppercase; letter-spacing:0.1em; font-weight:600;
                    color:var(--df-muted);">Model</div>
        <code style="font-size:0.78rem; color:var(--df-fg);">DiffusionGemma-26B-A4B-it</code>
      </div>
    </div>
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

    Both `start_pos` and `steps` accept either a list (one entry per target) or a
    scalar (broadcast to every target). `steps` defaults to all-zeros when absent
    or null.
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

    n = len(targets_v)

    def _broadcast(name: str, val, default):
        """Accept list-of-len-n, scalar (broadcast), or None (default-broadcast)."""
        if val is None:
            return [default] * n, None
        if isinstance(val, list):
            if len(val) == 1:
                return [val[0]] * n, None
            if len(val) != n:
                return None, f"`{name}` length {len(val)} does not match `targets` length {n}"
            return val, None
        # scalar (int/float) -> broadcast
        return [val] * n, None

    sp_list, err = _broadcast("start_pos", start_pos_v, 0)
    if err:
        return False, err
    steps_list, err = _broadcast("steps", payload.get("steps"), 0)
    if err:
        return False, err

    try:
        new_rows = [
            {"target": str(tgt), "start_pos": int(sp), "step": int(st_), "prob": 0.0}
            for tgt, sp, st_ in zip(targets_v, sp_list, steps_list)
        ]
    except (TypeError, ValueError) as exc:
        return False, f"could not coerce start_pos/steps to ints: {exc}"

    st.session_state["prompt_text"] = prompt_v
    # Drive the text_area through its own widget key directly. Just popping
    # `_prompt_widget` and relying on the `value=` fallback is unreliable
    # (Streamlit silently sticks with the previous widget value when the
    # widget renders on the next rerun) — assigning the key BEFORE the widget
    # is rendered is the supported way to push a new value into it.
    st.session_state["_prompt_widget"] = prompt_v
    st.session_state["rows"] = new_rows
    st.session_state["rows_nonce"] += 1
    return True, f"loaded {n} target(s) at positions {sp_list}"


def _normalize_decoded(decoded: list[dict]) -> list[dict]:
    """Restore int position-dict keys after a JSON round-trip (JSON keys are strings).

    The convergence views look records up with `rec["positions"].get(int_pos)`, so the
    string keys produced by `json.dump`/`json.load` would silently miss every lookup.
    """
    for rec in decoded:
        if isinstance(rec.get("positions"), dict):
            rec["positions"] = {int(k): v for k, v in rec["positions"].items()}
        if isinstance(rec.get("pre_positions"), dict):
            rec["pre_positions"] = {int(k): v for k, v in rec["pre_positions"].items()}
        if "steered_positions" in rec:
            rec["steered_positions"] = [int(p) for p in rec["steered_positions"]]
    return decoded


def _load_run_for_viz(payload: dict) -> tuple[bool, str]:
    """Load a saved-run JSON (a `frontend_runs/*.json` payload, or any dict with a
    `trace`) into `last_run` so the Results/Convergence tabs can visualize it."""
    if not isinstance(payload, dict):
        return False, "top-level JSON must be an object"
    decoded = payload.get("trace")
    if not isinstance(decoded, list) or not decoded:
        return False, "JSON has no `trace` records to visualize"
    decoded = _normalize_decoded(decoded)
    st.session_state["last_run"] = {
        "prompt": payload.get("prompt", ""),
        "baseline": payload.get("baseline", ""),
        "steered": payload.get("steered", ""),
        "landed": payload.get("landed", ""),
        "positions": payload.get("positions", []),
        "all_held": payload.get("all_held", False),
        "interventions": payload.get("interventions", []),
        "decoded": decoded,
        "trace_positions": payload.get("trace_positions", []),
        "trace_topk": int(payload.get("trace_topk", 8)),
        "config": payload.get("config", {}),
    }
    return True, f"loaded {len(decoded)} trace records"


def _form_is_dirty() -> bool:
    """True if the user has typed anything that would be lost by an import.

    "Dirty" means the prompt differs from the default OR any target row holds something
    other than the empty-default row. Used to decide whether to ask before clobbering.
    """
    if st.session_state.get("prompt_text", defaults.prompt) != defaults.prompt:
        return True
    rows = st.session_state.get("rows", [])
    if len(rows) != 1:
        return True
    return rows[0] != DEFAULT_ROW


def _reset_form_state() -> None:
    """Wipe the prompt + rows + last-run + uploader nonce back to defaults."""
    st.session_state["prompt_text"] = defaults.prompt
    # Push the default into the widget key directly so the text_area picks it up
    # on the next rerun (popping alone doesn't reliably update the widget).
    st.session_state["_prompt_widget"] = defaults.prompt
    st.session_state["rows"] = [dict(DEFAULT_ROW)]
    st.session_state["rows_nonce"] = st.session_state.get("rows_nonce", 0) + 1
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
    st.session_state.pop("last_run", None)
    st.session_state.pop("pending_import", None)


def _clear_setup_only() -> None:
    """Reset prompt + targets but keep last_run and the run log."""
    st.session_state["prompt_text"] = defaults.prompt
    st.session_state["_prompt_widget"] = defaults.prompt
    st.session_state["rows"] = [dict(DEFAULT_ROW)]
    st.session_state["rows_nonce"] = st.session_state.get("rows_nonce", 0) + 1
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
    st.session_state.pop("pending_import", None)


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.button(
        "🗑 Clear setup",
        on_click=_clear_setup_only,
        use_container_width=True,
        help="Reset prompt and targets back to defaults. Last run and log are kept.",
    )
    if st.button("➕ New run", type="primary", use_container_width=True,
                 help="Switch to Setup tab. Your prompt and targets are kept."):
        st.session_state["active_tab"] = "Setup"
        st.rerun()
    if st.button("🧹 Clear & start over", use_container_width=True,
                 help="Reset everything — prompt, targets, last run — back to defaults."):
        _reset_form_state()
        st.session_state["active_tab"] = "Setup"
        st.rerun()

    if "last_run" in st.session_state:
        st.caption(
            f"Last run: **{len(st.session_state['last_run'].get('decoded', []))}** trace records"
        )

    with st.expander("📊 Visualize saved run", expanded=False):
        st.caption(
            "Load a saved run JSON to scrub its denoising trace in the **Convergence** tab "
            "(green = how likely the top token is, blue = injected/pinned)."
        )
        _runs_dir = os.path.join(os.path.dirname(__file__), "frontend_runs")
        _files = (
            sorted(glob.glob(os.path.join(_runs_dir, "*.json")), key=os.path.getmtime, reverse=True)
            if os.path.isdir(_runs_dir) else []
        )
        viz_choice = st.selectbox(
            "from frontend_runs/", ["—"] + [os.path.basename(f) for f in _files],
        )
        viz_upload = st.file_uploader("…or upload a JSON", type=["json"], key="viz_uploader")
        if st.button(
            "📈 Load for visualization", use_container_width=True,
            disabled=(viz_choice == "—" and viz_upload is None),
        ):
            try:
                if viz_upload is not None:
                    payload = json.load(viz_upload)
                else:
                    with open(os.path.join(_runs_dir, viz_choice)) as _vf:
                        payload = json.load(_vf)
                ok, msg = _load_run_for_viz(payload)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"failed to load: {exc}"
            if ok:
                st.session_state["_pending_tab"] = "Convergence"
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

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
        max_new_tokens = st.number_input(
            "max_new_tokens", min_value=64, max_value=2048, value=512, step=64,
            help="Maximum number of tokens the model generates. Increase if responses are cut off.",
        )
        trace = st.checkbox(
            "--trace  (record per-step top-k for the convergence view)", value=True,
        )
        trace_all = st.checkbox(
            "trace ALL tokens (full canvas)", value=True,
            help="Record every canvas position each step so the convergence view shows the "
                 "whole sequence converging, not just the steered tokens. Heavier but needed "
                 "for the 'all tokens' view. Uncheck to record only the steered positions.",
        )
        trace_topk = st.number_input(
            "trace topk", min_value=1, value=defaults.trace_topk, step=1,
            help="server-side per-step top-k width when --trace is on",
        )
        trace_positions_text = st.text_input(
            "--trace-positions  (optional; extra positions when not tracing all)",
            value="", help="space-separated extra canvas positions to record",
        )

    run_log = st.session_state.get("run_log", [])
    if run_log:
        with st.expander(f"📋 Run log ({len(run_log)})", expanded=False):
            for i, entry in enumerate(reversed(run_log)):
                idx = len(run_log) - i
                held = "✅" if entry["all_held"] else "⚠️"
                st.markdown(
                    f"**#{idx}** {held} `{entry['landed']!r}`  \n"
                    f"<span style='font-size:0.78rem;color:var(--df-muted)'>"
                    f"positions {entry['positions']} · "
                    f"{len(entry['decoded'])} trace records</span>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"Prompt #{idx}", expanded=False):
                    st.caption(entry["prompt"])
                st.divider()

# --- Top tab nav (styled buttons; programmatic switch on run) ---------------
TABS = ["Setup", "Results", "Convergence"]
if "_pending_tab" in st.session_state:
    st.session_state["active_tab"] = st.session_state.pop("_pending_tab")
# A `?focus=N` in the URL means the user just clicked a token in the canvas and
# the page reloaded — auto-route them to the Convergence tab so they see the
# distribution for the position they clicked, not the Setup form.
if st.query_params.get("focus") is not None:
    st.session_state["active_tab"] = "Convergence"
active_tab = st.session_state.get("active_tab", "Setup")

# The tab buttons are scoped via their explicit `key="tab_btn_<name>"`; the matching
# CSS above (`div.st-key-tab_btn_*`) restyles them as a calm pill-strip independent of
# page-level primary buttons.
tab_cols = st.columns(len(TABS))
for col, tab in zip(tab_cols, TABS):
    if col.button(
        tab,
        use_container_width=True,
        type="primary" if tab == active_tab else "secondary",
        key=f"tab_btn_{tab}",
    ):
        st.session_state["active_tab"] = tab
        st.rerun()

st.divider()

# Pull rows + nonce -- both sit in session_state and are always available.
rows = st.session_state["rows"]
nonce = st.session_state["rows_nonce"]

# --- SETUP TAB -------------------------------------------------------------
if active_tab == "Setup":
    with st.expander("📂 Load experiment from JSON", expanded=False):
        st.caption(
            "Upload or paste a Simon's-style experiment JSON (e.g. files in "
            "`simons_experiments/`); the **prompt** and **targets** below are filled in. "
            "If the form already has work in it, you'll be asked to confirm before "
            "overwriting."
        )
        load_tab_upload, load_tab_paste = st.tabs(["📁 Upload file", "📋 Paste JSON"])

        # Stash the candidate payload here; if the form is dirty we render the confirm
        # buttons next to the controls and only commit on click.
        def _stage_import(payload: dict, source_label: str) -> None:
            """Validate then either apply immediately or stash for confirmation."""
            ok, msg = (True, "ok") if isinstance(payload, dict) else (False, "not an object")
            if not ok:
                st.error(f"⚠ {source_label}: {msg}")
                return
            if _form_is_dirty():
                st.session_state["pending_import"] = {
                    "payload": payload, "source": source_label,
                }
                st.rerun()
            else:
                ok, msg = _load_experiment_into_state(payload)
                if ok:
                    st.success(f"✅ {source_label} -- {msg}")
                    st.rerun()
                else:
                    st.error(f"⚠ {source_label}: {msg}")

        with load_tab_upload:
            uploaded = st.file_uploader(
                "experiment file", type=["json"], label_visibility="collapsed",
                key=f"exp_uploader_{st.session_state.get('uploader_nonce', 0)}",
            )
            if uploaded is not None:
                try:
                    payload = json.loads(uploaded.read().decode("utf-8"))
                    _stage_import(payload, uploaded.name)
                except json.JSONDecodeError as e:
                    st.error(f"could not parse JSON: {e}")

        with load_tab_paste:
            pasted = st.text_area(
                "paste JSON here",
                height=140, label_visibility="collapsed",
                placeholder='{"prompt": "...", "targets": ["..."], "start_pos": [0]}',
                key=f"exp_paste_{st.session_state.get('uploader_nonce', 0)}",
            )
            if st.button("Import pasted JSON", use_container_width=True):
                if not pasted.strip():
                    st.warning("Paste a JSON payload first.")
                else:
                    try:
                        payload = json.loads(pasted)
                        _stage_import(payload, "pasted JSON")
                    except json.JSONDecodeError as e:
                        st.error(f"could not parse JSON: {e}")

        # Confirmation prompt: "this will overwrite your current prompt + N target(s)".
        pending = st.session_state.get("pending_import")
        if pending is not None:
            cur_rows = len(st.session_state.get("rows", []))
            st.warning(
                f"Importing **{pending['source']}** will **clear your current prompt and "
                f"{cur_rows} target row(s)**. Continue?"
            )
            cc1, cc2, _ = st.columns([1, 1, 4])
            if cc1.button("✓ Clear & import", type="primary", use_container_width=True):
                payload = pending["payload"]
                _reset_form_state()
                ok, msg = _load_experiment_into_state(payload)
                st.session_state.pop("pending_import", None)
                if ok:
                    st.success(f"✅ {pending['source']} -- {msg}")
                else:
                    st.error(f"⚠ {pending['source']}: {msg}")
                st.rerun()
            if cc2.button("✕ Cancel", use_container_width=True):
                st.session_state.pop("pending_import", None)
                st.rerun()

        # Always-available "wipe the form" button outside the dirty path.
        st.button(
            "🆕 Clear all (reset prompt + targets + last run)",
            on_click=_reset_form_state, use_container_width=True,
        )

    st.markdown("#### Prompt")
    # Initialize the widget's session_state key on first render; thereafter the
    # widget owns the value and `_prompt_widget` is the source of truth. Imports
    # / resets write directly to `_prompt_widget` (see `_load_experiment_into_state`
    # and `_reset_form_state`) so the displayed prompt stays in sync.
    st.session_state.setdefault("_prompt_widget", st.session_state["prompt_text"])
    st.text_area(
        "--prompt", height=90, label_visibility="collapsed",
        key="_prompt_widget",
        on_change=lambda: st.session_state.update(
            prompt_text=st.session_state["_prompt_widget"]
        ),
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
            help="0 = hard pin (k=1); 0<p<=1 = soft pin, target gets prob p with residual "
                 "spread top-k proportionally (per-row, mixed values OK).",
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

# Catch the silent soft-pin-overridden case: TopKProportionalPolicy collapses to a hard
# point mass when k <= 1 OR p >= 1.0 (see steering/policies.py). So a row with 0<p<1
# only behaves as a soft pin if --k in the Advanced panel is also >=2; otherwise the
# requested probability is silently ignored. Warn loudly when the user has clearly asked
# for a soft pin but k=1 is going to clobber it.
_soft_rows = [(i, p) for i, p in enumerate(probs_per_target) if 0.0 < p < 1.0]
if _soft_rows and int(k) <= 1:
    st.warning(
        f"Soft `prob` set on row(s) {[i + 1 for i, _ in _soft_rows]} "
        f"({', '.join(f'p={p:.2f}' for _, p in _soft_rows)}), but `--k` is {int(k)} in "
        "Advanced. The policy collapses to a hard point mass when k≤1, so the soft prob "
        "will be silently ignored. Set `--k` ≥ 2 in the Advanced panel (e.g. 5) to get the "
        "top-k proportional residual you asked for."
    )

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
    submitted = st.button("Run experiment", type="primary", use_container_width=True)
else:
    submitted = False

# ---------------------------------------------------------------------------
# Run. Triggered by the sidebar button regardless of the active tab.
# ---------------------------------------------------------------------------

if submitted:
    if not targets:
        st.error("Need at least one target row with a non-empty target string.")
        st.stop()

    tokenizer = _tokenizer()
    where = {"host": host, "port": int(port)}

    # Per-target prob expanded to per-token list. The frontend treats prob==0.0 as the
    # "hard pin" sentinel (matching SteerConfig: prob=None means hard pin) and honors any
    # non-zero value as a literal probability. Mixed rows (some hard, some soft) work
    # because the build_interventions path accepts per-token None entries.
    if any(p > 0 for p in probs_per_target):
        per_token_probs: list[float | None] = []
        for tgt, p in zip(targets, probs_per_target):
            ids = tokenizer.encode(tgt, add_special_tokens=False)
            per_token_probs.extend([(p if p > 0 else None)] * len(ids))
        probabilities_arg: list[float | None] | None = per_token_probs
    else:
        probabilities_arg = None

    with st.spinner("Calling server..."):
        try:
            base = steer_call(
                prompt, tokens=[], positions=[], seed=int(seed),
                max_new_tokens=int(max_new_tokens), **where
            )
            steered_positions: list[int] = []
            for tgt, sp in zip(targets, start_pos):
                ids = tokenizer.encode(tgt, add_special_tokens=False)
                steered_positions.extend(range(sp, sp + len(ids)))
            tp = sorted(set(steered_positions) | set(trace_positions or []))
            # "all" tells the recorder to capture the whole canvas (every token), which is
            # what the convergence view needs; otherwise just the steered/explicit positions.
            trace_arg = "all" if trace_all else tp

            result = steer_strings(
                prompt, targets, start_pos, tokenizer,
                probabilities=probabilities_arg,
                ks=int(k),
                modes=modes, steps=steps,
                trace=bool(trace), trace_topk=int(trace_topk),
                trace_positions=trace_arg,
                seed=int(seed),
                max_new_tokens=int(max_new_tokens),
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
    # Append a compact summary to the persistent run log shown in the sidebar.
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []
    st.session_state["run_log"].append({
        "prompt": prompt,
        "landed": landed,
        "positions": result["positions"],
        "all_held": result["all_held"],
        "decoded": decoded,
        "baseline": base["text"],
        "steered": result["text"],
    })
    # Save full run to frontend_runs/ with a timestamp-based filename.
    _runs_dir = os.path.join(os.path.dirname(__file__), "frontend_runs")
    os.makedirs(_runs_dir, exist_ok=True)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _run_file = os.path.join(_runs_dir, f"run_{_ts}.json")
    _run_payload = {
        "timestamp": _ts,
        "prompt": prompt,
        "config": {
            "targets": targets, "start_pos": start_pos,
            "modes": modes, "steps": steps,
            "k": int(k), "prob": probs_per_target, "seed": int(seed),
        },
        "baseline": base["text"],
        "steered": result["text"],
        "landed": landed,
        "positions": result["positions"],
        "all_held": result["all_held"],
        "interventions": result["interventions"],
        "trace_positions": tp,
        "trace_topk": int(trace_topk),
        "trace": decoded,
    }
    with open(_run_file, "w") as _f:
        json.dump(_run_payload, _f, indent=2)
    # Auto-switch to the Results tab on a successful run.
    st.session_state["_pending_tab"] = "Results"
    st.rerun()

# If the page just reloaded after a token click in the Convergence canvas, the
# session is fresh — restore the most-recent saved run from disk so the click
# round-trip looks instant rather than wiping the screen.
_autoload_last_run()
last = st.session_state.get("last_run")

# ---------------------------------------------------------------------------
# Results + Convergence tabs (rendered when their tab is active).
# ---------------------------------------------------------------------------

if active_tab == "Results":
    if last is None:
        st.info("No run yet. Fill in inputs above and click **Run experiment**.")
    else:
        st.markdown(
            "<div style='text-align:right;color:var(--df-muted);font-size:0.85rem;"
            "padding-bottom:0.4rem'>"
            f"trace records: <b>{len(last.get('decoded') or [])}</b> · "
            f"steered positions: <b>{len(last['positions'])}</b></div>",
            unsafe_allow_html=True,
        )

        st.divider()
        c1, c2 = st.columns(2)
        c1.markdown(
            f"<div style='background:var(--df-card-bg);border:1px solid var(--df-card-border);"
            f"border-radius:8px;padding:0.4rem 0.8rem'>"
            f"<div style='font-size:0.78rem;color:var(--df-muted)'>Pinned positions</div>"
            f"<div style='font-size:1.1rem;font-weight:600;color:var(--df-fg)'>"
            f"{len(last['positions'])}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        # Held-state pill keeps a saturated tint in both themes; we just darken the
        # background and lighten the text in dark mode for readability.
        if last["all_held"]:
            held_bg = "light-dark(#dcfce7, #14532d)"
            held_color = "light-dark(#166534, #bbf7d0)"
            held_text = "✅ yes"
        else:
            held_bg = "light-dark(#fef3c7, #78350f)"
            held_color = "light-dark(#92400e, #fde68a)"
            held_text = "⚠️ no"
        c2.markdown(
            f"<div style='background:{held_bg};border:1px solid var(--df-card-border);"
            f"border-radius:8px;padding:0.4rem 0.8rem'>"
            f"<div style='font-size:0.78rem;color:var(--df-muted)'>All pins held?</div>"
            f"<div style='font-size:1.1rem;font-weight:600;color:{held_color}'>{held_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
        with st.expander(f"**Landed text** — `{last['landed']!r}`", expanded=False):
            st.code(last["landed"], language=None)

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
        st.caption(
            "If `all_held` is ✅, the attack stuck verbatim. If ⚠️, the model overrode "
            "at least one pin -- inspect the raw interventions table below to see which."
        )

        with st.expander("Raw interventions"):
            iv_df = enrich_interventions(last["interventions"], last["decoded"])
            st.caption(
                "`p` and `k` are what you **asked for** (the steer's requested prob and "
                "top-k width) -- that's why they're constant. The `nat_*` columns come from "
                "the trace and tell you what the **model** thought: "
                "`nat_top1_token_final` / `nat_top1_prob_final` = what it would have picked "
                "naturally at the last step and how confident it was; "
                "`nat_prob_of_requested_final` = the natural probability it assigned to the "
                "token we pinned (low → we forced something the model didn't believe in); "
                "`nat_top1_prob_first` / `_max` = how confidence at this position evolved. "
                "`post_top1_prob_final` is post-intervention top-1 prob (≈1.0 for held pins)."
            )
            st.dataframe(iv_df, use_container_width=True)

        st.divider()
        enriched = enrich_interventions(last["interventions"], last["decoded"])
        payload = {
            "prompt": last["prompt"],
            "config": last["config"],
            "baseline": last["baseline"],
            "steered": last["steered"],
            "landed": last["landed"],
            "positions": last["positions"],
            "all_held": last["all_held"],
            "interventions": enriched.to_dict(orient="records"),
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

        # Initial prompt card -- shown above the step slider so the user always has
        # the source prompt in view while scrubbing through the denoising trajectory.
        prompt_text = (last.get("prompt") or "").strip()
        if prompt_text:
            st.markdown(
                "<div style='background:var(--df-card-bg);border:1px solid var(--df-card-border);"
                "border-radius:8px;padding:0.5rem 0.85rem;margin:0.4rem 0 0.6rem 0'>"
                "<div style='font-size:0.78rem;color:var(--df-muted);"
                "text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.25rem'>"
                "Initial prompt</div>"
                "<div style='font-family:monospace;font-size:0.92rem;color:var(--df-fg);"
                "white-space:pre-wrap;word-break:break-word'>"
                f"{html.escape(prompt_text)}"
                "</div></div>",
                unsafe_allow_html=True,
            )

        slider_step = st.number_input(
            "Slider step size", min_value=1, max_value=int(all_steps[-1]),
            value=st.session_state.get("slider_step_val", 5), step=1,
            help="How many denoising steps the slider jumps per tick.",
            key="slider_step_val",
        )

        # Wrap the slider-driven view in a fragment so the step slider and
        # commitment-threshold slider rerun ONLY this block instead of the whole
        # app — that's what makes scrubbing feel live.
        @st.fragment
        def _convergence_view():
            # Anchor target so the post-click rerun keeps the canvas in view rather than
            # scrolling back to the top of the page.
            st.markdown("<div id='canvas-anchor'></div>", unsafe_allow_html=True)

            if "step_widget" not in st.session_state:
                st.session_state["step_widget"] = int(all_steps[-1])
            elif not (int(all_steps[0]) <= st.session_state["step_widget"] <= int(all_steps[-1])):
                st.session_state["step_widget"] = int(all_steps[-1])
            step = st.slider(
                "denoising step",
                min_value=int(all_steps[0]), max_value=int(all_steps[-1]),
                step=int(st.session_state.get("slider_step_val", 5)), key="step_widget",
            )

            st.markdown(f"##### Canvas at step {step}")
            canvas_html = _step_canvas_html(decoded, step, all_positions, steered_set)
            _render_canvas_iframe(
                canvas_html,
                height=820,
                frame_style=(
                    "border:1px solid #e3e3e3;border-radius:8px;padding:18px;"
                    "background:#fff;font-family:monospace;font-size:18px;"
                    "line-height:1.9;color:#111;"
                    "color-scheme:light dark;"
                ),
                scrolling=True,
            )
            st.caption(
                "Each glyph is the most-likely token at one position. "
                "<span style='background:rgb(220,245,230);color:#111;padding:0 3px;"
                "border-radius:3px'>"
                "Green</span> intensity ∝ its probability — pale = uncertain, "
                "<span style='background:rgb(22,163,74);color:#fff;padding:0 3px;border-radius:3px'>"
                "super green</span> = committed (p≈1). "
                "<span style='background:rgb(37,99,235);color:#fff;padding:0 3px;border-radius:3px'>"
                "Blue</span> = injected/pinned (dashed border = steered this very step).",
                unsafe_allow_html=True,
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
                    lines = (
                        alt.Chart(rk)
                        .mark_line(interpolate="step-after")
                        .encode(
                            x=alt.X("step:Q", title="denoising step"),
                            y=alt.Y("rank:Q", title="rank of final-winning token (1 = top)",
                                    scale=alt.Scale(reverse=True)),
                            detail="position:N",
                            color=alt.Color("position:N", legend=None),
                            tooltip=["step", "position", "rank"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(lines, use_container_width=True)
                    st.caption(
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
                "sampled step. Green intensity ∝ the top token's probability (pale = uncertain, "
                "super green = committed). Blue = a steered/pinned position; dashed blue border "
                "= actively steered at that exact step. Hover any token for its probability and, "
                "for steered steps, what the model would have picked naturally vs. what was forced."
            )
            stride = max(1, len(all_steps) // 24)
            sampled = all_steps[::stride]
            if all_steps[0] not in sampled:
                sampled.insert(0, all_steps[0])
            if all_steps[-1] not in sampled:
                sampled.append(all_steps[-1])
            first_step = all_steps[0]
            rows_html = []
            for s in sampled:
                canvas = _step_canvas_html(decoded, s, all_positions, steered_set)
                highlight = "background:#fff7d6;" if s == step else ""
                if s == first_step:
                    step_label = (
                        f"<span style='font-weight:700;color:#4f46e5'>step {s:>3}</span>"
                        f"<span style='display:block;font-size:9px;color:#a5b4fc;"
                        f"letter-spacing:0.04em'>start</span>"
                    )
                else:
                    step_label = f"step {s:>3}"
                rows_html.append(
                    f"<div style='display:flex;align-items:center;gap:10px;"
                    f"padding:4px 6px;border-bottom:1px solid #eee;"
                    f"{highlight}font-family:monospace;font-size:13px'>"
                    f"<div style='width:64px;color:#888;font-size:11px'>"
                    f"{step_label}</div>"
                    f"<div>{canvas}</div></div>"
                )
            # Film-strip lives in the same iframe as the main canvas so token clicks
            # there route through the same postMessage → ?focus=N → rerun pipeline.
            # Generous height; iframe scrolls internally via the wrapping `.df-frame`.
            _render_canvas_iframe(
                "".join(rows_html),
                height=min(520, 36 * len(sampled) + 40),
                frame_style=(
                    "border:1px solid #e3e3e3;border-radius:6px;padding:4px;"
                    "background:#fafafa;max-height:480px;overflow-y:auto;"
                    "color-scheme:light dark;"
                ),
            )

        _convergence_view()
