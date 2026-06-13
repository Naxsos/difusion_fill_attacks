"""Shared DiffusionGemma inference server.

Loads DiffusionGemma ONCE into the GPU, then serves prompts over HTTP so several
people can share a single A100 (the model is ~52 GB -- it only fits once).

Run on the box with the GPU:
    .venv/bin/python server.py                 # listens on 0.0.0.0:8000
    .venv/bin/python server.py --port 9000     # custom port

Then anyone can hit it (see client.py, or curl):
    curl -s localhost:8000/generate \
        -H 'content-type: application/json' \
        -d '{"prompt": "Why is the sky blue?", "max_new_tokens": 128}'

It also exposes the steering library (see steering/) at POST /steer, so interventions
reuse this already-loaded model instead of loading a second ~52 GB copy:
    curl -s localhost:8000/steer \
        -H 'content-type: application/json' \
        -d '{"prompt": "Count up: 1, 2, 3,", "tokens": ["7"], "positions": [5],
             "probabilities": 0.95, "ks": 2, "modes": "pin"}'

Concurrency model
-----------------
A single CUDA model can't run `generate` from multiple threads at once, so all
generation is serialized behind a lock. The HTTP server itself is threaded, so
extra requests simply queue (and report their queue wait) rather than failing.
For 3 users sending occasional prompts this is plenty; it is NOT a high-QPS setup.
"""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from load_model import load_model
from steering import RecorderConfig, steer

# Populated once in main(), shared (read-only) across request threads.
MODEL = None
PROCESSOR = None
# Serializes GPU work: only one generate() touches the model at a time.
GPU_LOCK = threading.Lock()


@torch.no_grad()
def run_generate(prompt: str, max_new_tokens: int = 128) -> str:
    """Tokenize one prompt, run the diffusion generate loop, return decoded reply."""
    message = [{"role": "user", "content": prompt}]
    inputs = PROCESSOR.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(MODEL.device)

    output = MODEL.generate(**inputs, max_new_tokens=max_new_tokens)

    # .generate returns a DiffusionGemmaGenerationOutput; .sequences includes the prompt.
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = output.sequences[0][prompt_len:]

    # DiffusionGemma generates in fixed 256-token canvases and rounds max_new_tokens UP
    # to a whole number of canvases, so it never returns fewer tokens than one canvas.
    # Trim to the exact request so max_new_tokens behaves as a real cap. (Note: a full
    # canvas is still computed under the hood, so a small cap isn't any faster.)
    new_tokens = new_tokens[:max_new_tokens]
    return PROCESSOR.decode(new_tokens, skip_special_tokens=True).strip()


def _serialize_trace(trace: list[dict] | None) -> list[dict] | None:
    """Turn the recorder's tensor-valued records into JSON-safe nested lists."""
    if trace is None:
        return None
    out = []
    for r in trace:
        rec = {k: r[k] for k in ("cur_step", "step_idx", "canvas_idx")}
        for key in ("argmax", "entropy", "positions", "topk_ids", "topk_probs", "logits"):
            if key in r:
                rec[key] = r[key].tolist()
        out.append(rec)
    return out


@torch.no_grad()
def run_steer(req: dict) -> dict:
    """Run one steered generation from a JSON request and return a JSON-safe payload.

    Interventions are given as parallel arrays (the `steer` API): `tokens` + `positions`
    are required, the rest broadcast. `include_trace` returns the per-step recorder data
    (can be large); `record` may be `true` or a RecorderConfig-shaped dict.
    """
    record = req.get("record", False)
    include_trace = bool(req.get("include_trace", False))
    if include_trace and not record:
        record = True
    if isinstance(record, dict):
        record = RecorderConfig(**record)

    result = steer(
        MODEL, PROCESSOR, req["prompt"],
        tokens=req.get("tokens"),
        positions=req.get("positions"),
        probabilities=req.get("probabilities"),
        steps=req.get("steps", 0),
        ks=req.get("ks", 2),
        modes=req.get("modes", "pin"),
        max_new_tokens=int(req.get("max_new_tokens", 256)),
        max_denoising_steps=req.get("max_denoising_steps"),
        keep_after_eos=bool(req.get("keep_after_eos", False)),
        suppress_eos_until=req.get("suppress_eos_until"),
        seed=req.get("seed"),
        record=record,
    )

    payload = {
        "text": result.text,
        "all_held": result.all_held,
        "interventions": [vars(o) for o in result.interventions],
    }
    if include_trace:
        payload["trace"] = _serialize_trace(result.trace)
    return payload


def _cuda_free_gb() -> dict:
    """Free / total VRAM on the current device, for callers managing model swaps."""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"free_gb": round(free / 1e9, 1), "total_gb": round(total / 1e9, 1)}


def run_unload() -> dict:
    """Drop the diffusion model and free its VRAM so another model can use the GPU.

    Lets a client (e.g. judge_experiments.py) temporarily borrow the whole card for a
    bigger judge model, then `/reload` the diffusion model afterwards. Runs under the
    GPU lock (held by the caller in do_POST), so no generation is in flight.
    """
    global MODEL, PROCESSOR
    was_loaded = MODEL is not None
    MODEL = None
    PROCESSOR = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"  /unload: released model (was_loaded={was_loaded})", flush=True)
    return {"unloaded": was_loaded, **_cuda_free_gb()}


def run_reload() -> dict:
    """Reload the diffusion model after an `/unload` (no-op if already loaded)."""
    global MODEL, PROCESSOR
    reloaded = MODEL is None
    if MODEL is None:
        t0 = time.perf_counter()
        MODEL, PROCESSOR = load_model()
        print(f"  /reload: model reloaded in {time.perf_counter() - t0:.0f}s", flush=True)
    return {"reloaded": reloaded, "loaded": MODEL is not None,
            "device": str(MODEL.device), **_cuda_free_gb()}


class Handler(BaseHTTPRequestHandler):
    # Quieter logs: one line per request is enough.
    def log_message(self, fmt, *args):  # noqa: A002 - matches stdlib signature
        print(f"  [{self.address_string()}] {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            loaded = MODEL is not None
            self._send_json(200, {
                "status": "ok",
                "model_loaded": loaded,
                "device": str(MODEL.device) if loaded else None,
                **_cuda_free_gb(),
            })
        else:
            self._send_json(404, {"error": f"unknown path {self.path!r}"})

    def _handle_generate(self, req: dict) -> dict:
        return {"response": run_generate(req["prompt"], int(req.get("max_new_tokens", 128)))}

    def _handle_steer(self, req: dict) -> dict:
        return run_steer(req)

    def _handle_unload(self, req: dict) -> dict:
        return run_unload()

    def _handle_reload(self, req: dict) -> dict:
        return run_reload()

    def do_POST(self) -> None:
        handlers = {
            "/generate": self._handle_generate,
            "/steer": self._handle_steer,
            "/unload": self._handle_unload,
            "/reload": self._handle_reload,
        }
        handler = handlers.get(self.path)
        if handler is None:
            self._send_json(404, {"error": f"unknown path {self.path!r}"})
            return

        try:
            length = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        # Measure how long we wait for the GPU vs. how long the work takes, so callers
        # can see when they're queued behind someone else.
        t_queued = time.perf_counter()
        with GPU_LOCK:
            t_start = time.perf_counter()
            try:
                payload = handler(req)
            except (KeyError, ValueError, TypeError) as exc:  # bad/missing fields
                self._send_json(400, {"error": f"bad request: {type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # surface model errors to the client
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            t_done = time.perf_counter()

        payload["generate_seconds"] = round(t_done - t_start, 2)
        payload["queue_seconds"] = round(t_start - t_queued, 2)
        self._send_json(200, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global MODEL, PROCESSOR
    print("Loading DiffusionGemma (full BF16, one-time ~3 min)...", flush=True)
    t0 = time.perf_counter()
    MODEL, PROCESSOR = load_model()
    print(f"Model loaded in {time.perf_counter() - t0:.0f}s on {MODEL.device}.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Serving on http://{args.host}:{args.port}  "
        "(POST /generate, /steer, /unload, /reload; GET /health)",
        flush=True,
    )
    print("Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
