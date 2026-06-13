"""Worked examples for talking to the shared DiffusionGemma server.

This is the client-side counterpart to examples.py: instead of loading the 52 GB
model itself, it sends prompts to a running server.py (which holds the one shared
copy on the A100). See SERVER.md for the full story.

Prerequisite: someone has started the server on the GPU box, e.g.
    .venv/bin/python server.py

Then run this from anywhere that can reach it (no torch/transformers needed):
    python example_client.py                       # talk to localhost:8000
    python example_client.py --host a100-box        # talk to a remote box

The examples below show the three patterns you'll actually use:
    1. one-shot prompts
    2. a multi-turn-ish batch of related prompts
    3. checking the server is alive before hammering it
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from client import generate

# Same sanity-check prompts as examples.py, so you can compare server vs. local.
PROMPTS: list[str] = [
    "Why is the sky blue? Answer in one sentence.",
    "Write a haiku about diffusion models.",
    "List three prime numbers greater than 50.",
    "Translate 'Good morning, how are you?' into French.",
]


def check_health(host: str, port: int) -> bool:
    """Return True if the server answers /health, else print why not and return False."""
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            info = json.loads(resp.read())
        print(f"Server is up on {info.get('device', '?')}.", flush=True)
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Could not reach server at {url}: {exc}")
        print("Is server.py running on the GPU box? See SERVER.md.")
        return False


def run_examples(host: str, port: int, max_new_tokens: int) -> None:
    """Send each prompt to the server and print the reply + timing."""
    print("\n" + "=" * 80)
    print(f"PROMPT / RESPONSE EXAMPLES  (server {host}:{port})")
    print("=" * 80)
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}] PROMPT: {prompt}", flush=True)
        t0 = time.perf_counter()
        response = generate(prompt, host=host, port=port, max_new_tokens=max_new_tokens)
        print(f"    RESPONSE ({time.perf_counter() - t0:.1f}s round-trip): {response}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="server port (default: 8000)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    # Fail fast with a friendly message if the server isn't there yet.
    if not check_health(args.host, args.port):
        return

    run_examples(args.host, args.port, args.max_new_tokens)


if __name__ == "__main__":
    main()
