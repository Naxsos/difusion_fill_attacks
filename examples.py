"""Worked examples for DiffusionGemma-26B-A4B-it.

Loads the model via the shared :func:`load_model.load_model`, prints the model
config, then runs a handful of prompts through the documented
chat-template -> generate -> decode path and prints the outputs.

Run:
    .venv/bin/python examples.py
"""

from __future__ import annotations

import json
import time

import torch

from load_model import load_model

# A few short, varied prompts to sanity-check the model end to end.
PROMPTS: list[str] = [
    "Why is the sky blue? Answer in one sentence.",
    "Write a haiku about diffusion models.",
    "List three prime numbers greater than 50.",
    "Translate 'Good morning, how are you?' into French.",
]


def print_config(model) -> None:
    """Print the model's config: a few key fields, then the full JSON."""
    config = model.config
    print("=" * 80)
    print("MODEL CONFIG")
    print("=" * 80)

    # Highlight the fields people usually care about (guarded — names vary by arch).
    highlights = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "vocab_size",
        "max_position_embeddings",
        "torch_dtype",
    ]
    for key in highlights:
        if hasattr(config, key):
            print(f"  {key}: {getattr(config, key)}")

    print("-" * 80)
    print("Full config:")
    # to_dict() -> JSON keeps it readable and complete.
    print(json.dumps(config.to_dict(), indent=2, default=str))
    print("=" * 80)


@torch.no_grad()
def generate(model, processor, prompt: str, max_new_tokens: int = 128) -> str:
    """Run one prompt through the chat template and return the decoded reply."""
    message = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the newly generated tokens (strip the prompt).
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = output[0][prompt_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    print("Loading DiffusionGemma (full BF16)...", flush=True)
    print(
        "  (one-time ~3 min: building the 26B mixture-of-experts graph, then "
        "streaming weights)",
        flush=True,
    )
    t0 = time.perf_counter()
    model, processor = load_model()
    print(f"Model loaded in {time.perf_counter() - t0:.0f}s.", flush=True)

    print_config(model)

    print("\n" + "=" * 80)
    print("PROMPT / RESPONSE EXAMPLES")
    print("=" * 80)
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}] PROMPT: {prompt}", flush=True)
        t0 = time.perf_counter()
        response = generate(model, processor, prompt)
        print(f"    RESPONSE ({time.perf_counter() - t0:.1f}s): {response}", flush=True)


if __name__ == "__main__":
    main()
