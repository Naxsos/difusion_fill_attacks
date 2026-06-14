#!/usr/bin/env python3
"""Report progress of a strong_reject.sh sweep.

A sweep writes one trace JSON per (prompt, strategy) job into siva_experiments/exp-N/,
named pXXXX_<mode>_<name>.json. With J jobs/prompt (from manifest.json) and P prompts
(from the prompt file), a full sweep produces P*J trace files; a prompt is "done" once
all J of its jobs have a trace file.

Usage:
    python run_status.py                 # latest exp-N
    python run_status.py exp-6           # a specific run (name or path)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "siva_experiments")
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strongreject_small.jsonl")


def latest_expdir() -> str:
    dirs = glob.glob(os.path.join(ROOT, "exp-*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        sys.exit(f"no exp-* dirs in {ROOT}")
    return max(dirs, key=lambda d: int(re.search(r"exp-(\d+)$", d).group(1)))


def resolve(arg: str) -> str:
    if os.path.isdir(arg):
        return arg
    cand = os.path.join(ROOT, arg)
    if os.path.isdir(cand):
        return cand
    sys.exit(f"no such exp dir: {arg}")


def jobs_per_prompt(expdir: str) -> tuple[int, list[str]]:
    """Read the manifest copied into the run to learn how many jobs each prompt has."""
    man = os.path.join(expdir, "manifest.json")
    if not os.path.exists(man):
        return 0, []
    m = json.load(open(man))
    runs = m["runs"] if "runs" in m else [{"mode": m.get("mode", "pin"), "strategies": m["strategies"]}]
    names = []
    for run in runs:
        mode = run.get("mode", "pin")
        for s in run["strategies"]:
            names.append(f"{mode}/{s['name']}")
    return len(names), names


def total_prompts() -> int:
    if not os.path.exists(PROMPT_FILE):
        return 0
    with open(PROMPT_FILE) as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    expdir = resolve(sys.argv[1]) if len(sys.argv) > 1 else latest_expdir()
    jpp, names = jobs_per_prompt(expdir)
    nprompts = total_prompts()

    # Each completed job leaves a pXXXX_*.json trace file (exclude manifest.json).
    traces = [p for p in glob.glob(os.path.join(expdir, "p*.json"))]
    per_prompt: dict[str, int] = {}
    for t in traces:
        m = re.match(r"p(\d{4})_", os.path.basename(t))
        if m:
            per_prompt[m.group(1)] = per_prompt.get(m.group(1), 0) + 1

    done_jobs = sum(per_prompt.values())
    done_prompts = sum(1 for c in per_prompt.values() if jpp and c >= jpp)
    started_prompts = len(per_prompt)
    total_jobs = nprompts * jpp if (nprompts and jpp) else 0

    graded = os.path.exists(os.path.join(expdir, "strongreject_scores.csv"))
    errfile = os.path.join(expdir, "errors.txt")
    nerr = sum(1 for _ in open(errfile)) if os.path.exists(errfile) else 0

    print(f"run:        {expdir}")
    print(f"strategies: {jpp} jobs/prompt -> {', '.join(names)}")
    print(f"prompts:    {done_prompts}/{nprompts} fully done"
          + (f"  ({started_prompts} started)" if started_prompts != done_prompts else ""))
    if total_jobs:
        pct = 100 * done_jobs / total_jobs
        print(f"jobs:       {done_jobs}/{total_jobs} traces written ({pct:.1f}%)")
    else:
        print(f"jobs:       {done_jobs} traces written")
    if nerr:
        print(f"errors:     {nerr} (see {errfile})")
    print(f"grading:    {'done (strongreject_scores.csv)' if graded else 'not yet'}")


if __name__ == "__main__":
    main()
