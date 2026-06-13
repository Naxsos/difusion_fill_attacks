"""Thin client for the shared DiffusionGemma server (see server.py).

No dependencies -- uses only the standard library, so it runs anywhere (your
laptop, another conda env, etc.) as long as it can reach the server host.

As a library:
    from client import generate
    print(generate("Why is the sky blue?", host="a100-box", port=8000))

From the shell:
    .venv/bin/python client.py "Write a haiku about diffusion models."
    .venv/bin/python client.py --host a100-box --max-new-tokens 256 "Explain MoE."
"""

from __future__ import annotations

import argparse
import json
import urllib.request

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000


def generate(
    prompt: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_new_tokens: int = 128,
    timeout: float = 600.0,
) -> str:
    """POST a prompt to the server and return the decoded reply."""
    url = f"http://{host}:{port}/generate"
    data = json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens}).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload["response"]


def steer(
    prompt: str,
    tokens: list,
    positions: list,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    probabilities=None,
    steps: int | list[int] = 0,
    ks: int | list[int] = 2,
    modes: str | list[str] = "pin",
    max_new_tokens: int = 256,
    keep_after_eos: bool = False,
    suppress_eos_until=None,
    seed: int | None = None,
    include_trace: bool = False,
    record=None,
    timeout: float = 600.0,
) -> dict:
    """POST a steered-generation request to the server's /steer endpoint.

    `tokens` (int ids or single-token strings) and `positions` are the interventions;
    the other knobs broadcast (see steering.steer). Returns the JSON payload, including
    `text`, `interventions` (with `held`), and `trace` if `include_trace=True`.

    `record` is an optional RecorderConfig-shaped dict (e.g.
    `{"positions": [0, 1], "top_k": 5}`) controlling *what* the per-step recorder
    captures; setting it implies a trace. `include_trace=True` returns that trace in the
    payload under `trace` (a list of per-denoising-step records).
    """
    if record is not None and not include_trace:
        include_trace = True
    url = f"http://{host}:{port}/steer"
    body = json.dumps({
        "prompt": prompt, "tokens": tokens, "positions": positions,
        "probabilities": probabilities, "steps": steps, "ks": ks, "modes": modes,
        "max_new_tokens": max_new_tokens, "keep_after_eos": keep_after_eos,
        "suppress_eos_until": suppress_eos_until,
        "seed": seed, "include_trace": include_trace, "record": record,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(generate(args.prompt, host=args.host, port=args.port, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
