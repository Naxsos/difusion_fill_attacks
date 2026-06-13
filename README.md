# diffusion_fill_attacks

Fill-attack experiments against Google's **DiffusionGemma-26B-A4B-it**, a discrete-diffusion MoE LLM.

## Setup

Create an isolated virtual environment and install the pinned dependencies:

```bash
cd hackathon_SST/diffusion_fill_attacks
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python load_model.py   # smoke test
```

## Loading the model

All scripts share one loader. Import it instead of calling `from_pretrained` directly:

```python
from load_model import load_model

model, processor = load_model()  # full BF16 (~52 GB) — needs an 80 GB A100/H100
```

`load_model()` returns `(model, processor)`. Key options: `dtype=...` (defaults to
`"auto"` → the checkpoint's native BF16), `device_map=...` (defaults to `"auto"`),
`model_id=...` to override the repo, `hf_token=...` (else falls back to the `HF_TOKEN`
env var; public model so it's optional).

> Quantization was removed: bitsandbytes can't touch this model's batched MoE expert
> tensors (the bulk of the 26B params), so 4-bit/8-bit stayed ~52 GB and just forced
> slow CPU offload. Run it in full BF16 on a big GPU instead.

Run the module directly for a smoke test:

```bash
python load_model.py
```

Requires a CUDA GPU with enough memory for the full BF16 model (~52 GB → an 80 GB
A100/H100) and a recent `transformers` (one that ships
`DiffusionGemmaForBlockDiffusion`).

## Sharing one GPU between several people

The model only fits **once** on an 80 GB A100, so don't have everyone call
`load_model()`. Instead load it once in [`server.py`](server.py) and have everyone
prompt it over HTTP via [`client.py`](client.py) / [`example_client.py`](example_client.py):

```bash
.venv/bin/python server.py          # on the GPU box: loads once, then serves
python example_client.py --host a100-box   # from anywhere: no torch needed
```

See [SERVER.md](SERVER.md) for the full guide (API, remote access, concurrency).

## Steering / intervening in the denoising loop

The [`steering/`](steering/) package forces a chosen token at a chosen output position,
with a chosen probability, at a chosen denoising step — and exposes per-step
logits/probabilities. It also plugs into the server as `POST /steer`. See
[STEERING.md](STEERING.md).

## Streamlit workbench

[`streamlit_app.py`](streamlit_app.py) is a frontend over the same `client.steer`
pipeline `example_steer.py` and `run_experiments.py` use — the heavy ~52 GB model
stays on the server, so the workbench only needs the tokenizer + UI deps:

```bash
pip install streamlit pandas
streamlit run streamlit_app.py
```

It gives you a form for every `SteerConfig` knob (one row per target so you can stage
multiple steers at different denoising steps), shows the baseline and steered text
side-by-side, lists what landed at each pinned position, and visualizes the
denoising-loop convergence from the per-step trace:

- top-1 token trajectory per traced position (one column per step),
- top-1 probability over denoising steps per position (line chart),
- top-k probability stack at a selected position (area chart) — watch competing
  tokens decay as the canvas commits.

Each run can be downloaded as a JSON identical in shape to the trace files
`example_steer --trace-file` writes, so it slots into the existing analysis tools.
