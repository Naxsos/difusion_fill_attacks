"""Run DiffusionGemma with GPU+CPU offload (no bnb).

bitsandbytes can't quantize the fused MoE expert tensors, so 4/8-bit doesn't
shrink this model meaningfully. Instead we load in native bf16 and let
accelerate keep ~38 GB on the GPU and spill the overflow (mostly experts) to
CPU RAM. Slower per token, but it actually fits and runs.
"""

from __future__ import annotations

import logging
import os

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

logging.basicConfig(level=logging.INFO)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

print("Loading in bf16 with GPU+CPU offload (one-time, ~minutes)...", flush=True)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    # Leave headroom on the 44 GB card for activations / KV cache; rest -> CPU.
    max_memory={0: "38GiB", "cpu": "300GiB"},
    token=os.environ.get("HF_TOKEN"),
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))

print("LOADED OK", flush=True)
print("device_map (head):", flush=True)
# Show how layers got split across GPU(0) vs cpu.
dm = getattr(model, "hf_device_map", {})
from collections import Counter
print("  placement counts:", Counter(str(v) for v in dm.values()), flush=True)

prompt = "Why is the sky blue? Answer in one sentence."
msg = [{"role": "user", "content": prompt}]
inputs = processor.apply_chat_template(
    msg, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to("cuda:0")

print(f"\nPROMPT: {prompt}\nGenerating...", flush=True)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
resp = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
print(f"RESPONSE: {resp}", flush=True)
