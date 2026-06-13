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
