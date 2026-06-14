#!/usr/bin/env python3
"""Pivot the long cumulative scores CSV into two wide, per-prompt files.

Input  (long):  strongreject_scores_cumulative.csv
                columns: prompt_idx, strategy, score, forbidden_prompt, response
Outputs (wide, one row per prompt):
  responses.xlsx : prompt, <strat1>, <strat2>, <strat3>   (cell = that strategy's response)
  scores.csv     : prompt, <strat1>, <strat2>, <strat3>   (cell = that strategy's [0,1] score)

Usage:
  python export_pivot.py [--src siva_experiments/exp-6/strongreject_scores_cumulative.csv]
                         [--outdir siva_experiments/exp-6]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

DEF_SRC = "siva_experiments/exp-6/strongreject_scores_cumulative.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEF_SRC)
    ap.add_argument("--outdir", default=None, help="defaults to the src's directory")
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.src))
    df = pd.read_csv(args.src)

    # one canonical prompt string per index (forbidden_prompt is constant per prompt_idx)
    prompts = (df.groupby("prompt_idx")["forbidden_prompt"].first())
    strategies = sorted(df["strategy"].unique())

    resp = (df.pivot_table(index="prompt_idx", columns="strategy",
                           values="response", aggfunc="first")
              .reindex(columns=strategies))
    score = (df.pivot_table(index="prompt_idx", columns="strategy",
                            values="score", aggfunc="first")
               .reindex(columns=strategies))

    resp.insert(0, "prompt", prompts)
    score.insert(0, "prompt", prompts)
    resp = resp.reset_index(drop=True)
    score = score.reset_index(drop=True)

    xlsx = os.path.join(outdir, "responses.xlsx")
    csv = os.path.join(outdir, "scores.csv")
    resp.to_excel(xlsx, index=False)
    score.to_csv(csv, index=False)

    print(f"strategies: {strategies}")
    print(f"prompts:    {len(resp)}")
    print(f"wrote {xlsx}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
