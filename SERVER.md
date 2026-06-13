# Shared inference server

DiffusionGemma is ~52 GB in BF16, so it only fits **once** on an 80 GB A100. If
several people each call `load_model()` they'll OOM. Instead, load the model once
in a long-running server ([`server.py`](server.py)) and have everyone talk to it
over HTTP ([`client.py`](client.py)).

Both files use only the Python standard library (`http.server` / `urllib`), so the
client runs anywhere — another conda env, a teammate's laptop — with no `torch` or
`transformers` install required.

## Starting the server

Run it on the box with the GPU, inside `tmux`/`screen` so it survives your SSH
session disconnecting:

```bash
cd difusion_fill_attacks
.venv/bin/python server.py              # listens on 0.0.0.0:8000
.venv/bin/python server.py --port 9000  # custom port
```

It loads the model once (one-time ~3 min), prints `Serving on http://...`, and then
stays up handling requests until you Ctrl-C it.

> **Don't run `examples.py` alongside the server** — that loads a *second* ~52 GB
> copy and you'll OOM. Once the server is up, route everything through it.

## Calling it

### Python (library)

```python
from client import generate

reply = generate("Why is the sky blue?", host="localhost", max_new_tokens=128)
print(reply)
```

### Command line

```bash
.venv/bin/python client.py "Write a haiku about diffusion models."
.venv/bin/python client.py --host a100-box --max-new-tokens 256 "Explain MoE routing."
```

### Raw curl

```bash
curl -s localhost:8000/generate \
    -H 'content-type: application/json' \
    -d '{"prompt": "Why is the sky blue?", "max_new_tokens": 128}'
```

## HTTP API

| Method | Path        | Body                                   | Returns |
|--------|-------------|----------------------------------------|---------|
| `GET`  | `/health`   | —                                      | `{"status": "ok", "model_loaded": bool, "device": "cuda:0", "free_gb": float, "total_gb": float}` |
| `POST` | `/generate` | `{"prompt": str, "max_new_tokens": int}` | `{"response": str, "generate_seconds": float, "queue_seconds": float}` |
| `POST` | `/unload`   | —                                      | `{"unloaded": bool, "free_gb": float, "total_gb": float}` |
| `POST` | `/reload`   | —                                      | `{"reloaded": bool, "loaded": true, "device": "cuda:0", ...}` |

`max_new_tokens` is optional (defaults to 128). `queue_seconds` reports how long the
request waited behind another generation — useful when two people fire at once.

### Model swap: `/unload` + `/reload`

`/unload` drops the diffusion model and frees its ~52 GB; `/reload` brings it back
(another ~3 min). This lets a client temporarily borrow the whole GPU for a bigger
model — e.g. [`judge_experiments.py --backend vllm`](judge_experiments.py) swaps the
diffusion model out for a ~61 GB `gpt-oss-120b` judge, then reloads it. Both run under
the GPU lock, so they can't race an in-flight `generate`. **Heads-up for shared use:**
`/unload` evicts the model for *everyone* on the server until `/reload` finishes, so
don't trigger the swap while teammates are mid-session.

## Reaching it from another machine

If your teammates aren't on the A100 box itself:

- **Same network:** point `--host` at the box's hostname or IP (the server binds
  `0.0.0.0`, so it accepts external connections).
- **SSH tunnel** (works through firewalls):
  ```bash
  ssh -L 8000:localhost:8000 a100-box   # leave this open
  # now, on your laptop:
  python client.py --host localhost "your prompt"
  ```

## Concurrency model

The HTTP server is threaded (accepts concurrent connections), but a single CUDA
model can't run `generate` from multiple threads at once — so all generation is
serialized behind a lock. Extra requests **queue** (and report `queue_seconds`)
rather than failing. This is correct and simple for ~3 people sending occasional
prompts; it is **not** a high-QPS setup. If you need better A100 utilization when
everyone's active at once, batch prompts that arrive close together into a single
`generate` call (DiffusionGemma supports a batch dimension).
