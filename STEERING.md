# Steering / intervention library

`steering/` lets you reach into DiffusionGemma's denoising loop and **force a chosen
token at a chosen output position, with a chosen probability, at a chosen denoising
step** — and read back the per-step logits/probabilities for the whole 256-token canvas.

It works by passing a custom `LogitsProcessor` to `model.generate`, which the model
calls every denoising step with the `(batch, 256, vocab)` logits for the current canvas.
That processor's output feeds sampling, the accept/renoise sampler, self-conditioning,
and (at the final step) the emitted token — so it's a deep, sufficient intervention
point. See [load_model.py](load_model.py) for loading and `difusion_fill_attacks/.venv`'s
`transformers/models/diffusion_gemma/generation_diffusion_gemma.py` for the loop itself.

## Quickstart (in-process)

```python
from load_model import load_model
from steering import steer

model, processor = load_model()

result = steer(
    model, processor, "Count up from one: 1, 2, 3,",
    tokens=["7"], positions=[5],     # put token "7" at the 6th generated token
    probabilities=0.95, ks=2,        # 95% on "7", 5% on the top runner-up
    steps=3, modes="pin",            # from step 3 through the end of denoising
    record=True,                     # also capture the per-step trace
)
print(result.summary())
print(result.all_held)               # did every intervention land?
trace = result.trace                 # per-step probs/argmax/entropy
```

## `steer(...)` parameters

| Param | Meaning |
|---|---|
| `tokens` | int vocab ids or **single-token** strings (multi-token strings are rejected). |
| `positions` | 0-indexed in the **generated** output (0 = first new token, not the prompt). |
| `probabilities` | mass `p` on the target token; scalar (broadcast) or per-token list. Default 1.0. |
| `ks` | `k=1` → hard point mass; `k>=2` → target gets `p`, residual `1−p` spread over the top `k−1` runner-ups ∝ the model's own probabilities. |
| `steps` | 0-indexed from the **start** of denoising (step 0 = first iteration). |
| `modes` | `"perturb"` (fire only at `step`) or `"pin"` (fire from `step` to the end). |
| `max_new_tokens` | rounded up to whole 256-token canvases by the model. |
| `max_denoising_steps` | fixed step count (defaults to the model's config, 48). |
| `disable_adaptive_stopping` | keep step indices stable/reproducible (default `True`). |
| `record` | `True`, a `RecorderConfig`, or `None`/`False`. |
| `seed` | seeds the multinomial sampling for reproducibility. |

Every per-token arg is either a scalar (broadcast across all positions) or a list the
same length as `tokens`/`positions`. You can also pass a prebuilt list via
`interventions=[...]` instead of the arrays.

## What is actually guaranteed

The emitted token at a position is the **argmax of the final denoising step's logits**,
and acceptance is re-decided every step. So:

- **`pin` + `k=1`/`p=1.0` (or any `p>0.5`)** → the final step's argmax is the target ⇒
  **output token guaranteed** (and `k=1`/`p=1.0` gives entropy 0, so it's accepted/frozen
  every pinned step).
- **`pin` + `p<0.5`** → output not guaranteed (a runner-up can be the argmax).
- **`perturb`** → a single-step nudge; whether it survives to the output is emergent —
  inspect `result.trace` to see how it propagates.

This is verified end-to-end in [steering/_smoke.py](steering/_smoke.py) (freeze accuracy,
exact 0.95 probability after temperature compensation, residual split, perturb-vs-pin,
recorder shapes).

## Reading the trace

With `record=True`, `result.trace` is a list (one dict per denoising step, in call
order). Each record has `cur_step`, `step_idx`, `canvas_idx`, plus tensors:
`argmax` `(256,)`, `entropy` `(256,)`, and `topk_ids`/`topk_probs`/`positions` for the
recorded positions. Customize via `RecorderConfig`:

```python
from steering import RecorderConfig
steer(..., record=RecorderConfig(positions=[5], top_k=10))   # only position 5, top-10
steer(..., record=RecorderConfig(record_full_logits=True))   # whole vocab (multi-GB!)
```

The recorder reports **post-temperature** probabilities by default (what the sampler
actually sees); set `post_temperature=False` for the raw injected logits.

## Server endpoint (no in-process load)

The library loads the model in-process, which can't coexist with the shared server (one
~52 GB copy per A100). To avoid that, [server.py](server.py) exposes `POST /steer`, which
reuses the already-loaded model:

```python
from client import steer
out = steer("Count up: 1, 2, 3,", tokens=["7"], positions=[5],
            probabilities=0.95, ks=2, modes="pin", host="a100-box",
            include_trace=False)
print(out["text"], out["all_held"])
```

Or curl:

```bash
curl -s localhost:8000/steer -H 'content-type: application/json' \
  -d '{"prompt":"Count up: 1, 2, 3,","tokens":["7"],"positions":[5],
       "probabilities":0.95,"ks":2,"modes":"pin"}'
```

Request fields mirror `steer(...)`; `include_trace: true` returns the (potentially large)
trace as nested lists. Like `/generate`, calls are serialized behind the GPU lock and
report `queue_seconds`/`generate_seconds`.

## Extending it (no core changes)

Three drop-in extension points — subclass and pass the instance in:

- **`DistributionPolicy`** ([steering/policies.py](steering/policies.py)) — *what*
  distribution to inject. Default `TopKProportionalPolicy`. Add uniform-residual,
  entropy-targeted, logit-bias, or suppression policies here.
- **`StepSchedule`** ([steering/schedule.py](steering/schedule.py)) — *when* to fire.
  `PerturbAt` / `PinFrom` ship; add ranges, every-other-step, etc.
- **`RecorderConfig` / `StepRecorder`** ([steering/processor.py](steering/processor.py))
  — *what* to capture each step.

## Notes & gotchas

- **Single-token only.** This tokenizer has a standalone space token, so `" 7"` is two
  tokens but `"7"` is one. Pass bare tokens or int ids; multi-token strings raise.
- **Temperature compensation.** The processor writes `T·log(D)` so that the model's
  built-in temperature schedule (`÷T`) recovers exactly `D` — that's why `p=0.95` comes
  out as 0.95 in the trace, not skewed by temperature.
- **Reproducibility.** Adaptive early-stopping is disabled by default so `step` indices
  are stable; pass `disable_adaptive_stopping=False` to restore the model's default
  behavior (then high `step` indices may be skipped).
