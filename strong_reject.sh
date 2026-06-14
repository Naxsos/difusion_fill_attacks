#!/usr/bin/env bash
set -euo pipefail

# --- activate virtualenv --------------------------------------------------
VENV="${VENV:-.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "no venv at $VENV/bin/activate — set VENV=path/to/.venv or create one" >&2
  exit 1
fi

# --- config ---------------------------------------------------------------
SCRIPT="python example_steer.py"     # your script's invocation
PROMPT_FILE="strongreject_small.jsonl"
MANIFEST="${MANIFEST:-./siva_experiments/manifest.json}"  # strategies live here now
MAX_PROMPTS="${MAX_PROMPTS:-0}"      # cap prompts processed (0 = all); override via env
START_PROMPT="${START_PROMPT:-0}"    # skip prompts with index < this (resume a partial run)
OUTDIR_OVERRIDE="${OUTDIR:-}"        # write into an existing dir instead of a fresh exp-N
GRADER="${GRADER:-strongreject_finetuned}"   # autograder evaluator
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-2b}" # ungated base for the finetuned grader

# --- strategies come from manifest.json -----------------------------------
# Each strategy = {name, positions[], targets[]}; we pin all spans from `step`
# in `mode`. Edit $MANIFEST to change the attack; this script no longer hardcodes
# strategies. A diffusion completion is a 256-token canvas filled JOINTLY, so a
# content-forcing lead-in that ends mid-sentence on a concrete verb phrase
# ("...First, you should") crowds out the refusal and forces specifics.
if [[ ! -f "$MANIFEST" ]]; then
  echo "no manifest at $MANIFEST — set MANIFEST=path/to/manifest.json" >&2
  exit 1
fi
US=$'\x1f'  # span sep   RS=$'\x1e' field sep   GS=$'\x1d' job sep

# Flatten runs[] x strategies into per-job records, each carrying its OWN
# mode/step (so one manifest can mix e.g. pin and perturb). Falls back to a
# single top-level run for manifests that predate the runs[] schema.
declare -a STRAT_MODES STRAT_STEPS STRAT_NAMES STRAT_POSITIONS STRAT_TARGETS
while IFS= read -r -d $'\x1d' rec; do
  [[ -z "$rec" ]] && continue
  mode="${rec%%$'\x1e'*}";   rest="${rec#*$'\x1e'}"
  step="${rest%%$'\x1e'*}";  rest="${rest#*$'\x1e'}"
  name="${rest%%$'\x1e'*}";  rest="${rest#*$'\x1e'}"
  pos="${rest%%$'\x1e'*}";   targets="${rest#*$'\x1e'}"
  STRAT_MODES+=("$mode"); STRAT_STEPS+=("$step"); STRAT_NAMES+=("$name")
  STRAT_POSITIONS+=("$pos"); STRAT_TARGETS+=("$targets")
done < <(python -c '
import json, sys
m = json.load(open(sys.argv[1]))
runs = m["runs"] if "runs" in m else [{"mode": m.get("mode","pin"), "step": m.get("step",0), "strategies": m["strategies"]}]
for run in runs:
    mode = run.get("mode", "pin"); step = run.get("step", 0)
    for s in run["strategies"]:
        sys.stdout.write(mode + "\x1e" + str(step) + "\x1e" + s["name"] + "\x1e"
                         + " ".join(str(p) for p in s["positions"]) + "\x1e"
                         + "\x1f".join(s["targets"]) + "\x1d")
' "$MANIFEST")
echo "manifest $MANIFEST -> ${#STRAT_NAMES[@]} jobs/prompt:"
for s in "${!STRAT_NAMES[@]}"; do
  echo "  ${STRAT_MODES[$s]}/${STRAT_NAMES[$s]} (step=${STRAT_STEPS[$s]}, pos=[${STRAT_POSITIONS[$s]}])"
done

# --- experiment dir: ./siva_experiments/exp-{no}, auto-incremented --------
mkdir -p ./siva_experiments
if [[ -n "$OUTDIR_OVERRIDE" ]]; then
  OUTDIR="$OUTDIR_OVERRIDE"
else
  n=1
  while [[ -d "./siva_experiments/exp-${n}" ]]; do n=$((n+1)); done
  OUTDIR="./siva_experiments/exp-${n}"
fi
mkdir -p "$OUTDIR"
cp "$MANIFEST" "$OUTDIR/manifest.json"   # record the exact strategies with the traces
echo "writing to $OUTDIR (start=$START_PROMPT, max=$MAX_PROMPTS)"

# --- fetch StrongREJECT-small at runtime ----------------------------------
# pulls the prompt subset from the public benchmark; does not embed prompts.
if [[ ! -f "$PROMPT_FILE" ]]; then
  CSV_URL="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv" \
  PROMPT_FILE="$PROMPT_FILE" \
  python - <<'PY'
import os
import pandas as pd
df = pd.read_csv(os.environ["CSV_URL"])
assert "forbidden_prompt" in df.columns, f"unexpected columns: {list(df.columns)}"
df.to_json(os.environ["PROMPT_FILE"], orient="records", lines=True, force_ascii=False)
print(f"loaded {len(df)} prompts")
PY
fi

# --- sweep: every prompt x every strategy ---------------------------------
i=0
while IFS= read -r line; do
  if [[ "$MAX_PROMPTS" -gt 0 && "$i" -ge "$MAX_PROMPTS" ]]; then
    echo "reached MAX_PROMPTS=$MAX_PROMPTS — stopping"
    break
  fi
  if [[ "$i" -lt "$START_PROMPT" ]]; then i=$((i+1)); continue; fi
  prompt=$(printf '%s' "$line" | python -c 'import sys,json; print(json.load(sys.stdin)["forbidden_prompt"])')
  pidx=$(printf '%04d' "$i")

  for s in "${!STRAT_NAMES[@]}"; do
    mode="${STRAT_MODES[$s]}"; step="${STRAT_STEPS[$s]}"; name="${STRAT_NAMES[$s]}"
    # split this strategy's spans on \x1f (newline-safe, unlike `read`)
    readarray -d "$US" -t targets < <(printf '%s' "${STRAT_TARGETS[$s]}")
    read -r -a positions <<< "${STRAT_POSITIONS[$s]}"
    # tag carries the mode so pin/multispan_list and perturb/multispan_list don't collide
    tag="p${pidx}_${mode}_${name}"
    $SCRIPT \
      --prompt "$prompt" \
      --target "${targets[@]}" \
      --start-pos "${positions[@]}" \
      --step "$step" \
      --mode "$mode" \
      --no-baseline \
      --trace \
      --trace-file "$OUTDIR/${tag}.json" \
      > "$OUTDIR/${tag}.log" 2>&1 || echo "FAILED: $tag" >> "$OUTDIR/errors.txt"
  done
  i=$((i+1))
done < "$PROMPT_FILE"

echo "sweep done — traces + logs in $OUTDIR"

# --- grade every response with the StrongREJECT autograder ----------------
# Writes ONE csv ($OUTDIR/strongreject_scores.csv) with both strategies'
# responses and their [0,1] harmfulness scores.
echo "grading with $GRADER ..."
python siva_experiments/strongreject_grade.py \
  --expdir "$OUTDIR" \
  --evaluator "$GRADER" \
  --base-model "$BASE_MODEL"

echo "done — scores in $OUTDIR/strongreject_scores.csv"
