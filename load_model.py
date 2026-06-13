"""Reusable loader for Google's DiffusionGemma-26B-A4B-it (a discrete-diffusion MoE LLM).

Other scripts in this project should import :func:`load_model` rather than calling
``from_pretrained`` themselves, so the whole project shares one canonical way of
loading the model + processor.

Example
-------
    from load_model import load_model

    model, processor = load_model()  # full BF16

    message = [{"role": "user", "content": "Why is the sky blue?"}]
    inputs = processor.apply_chat_template(
        message, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    print(processor.decode(out[0], skip_special_tokens=True))

Notes
-----
* DiffusionGemma loads via the custom ``DiffusionGemmaForBlockDiffusion`` class
  (NOT ``AutoModelForCausalLM``) and does not require ``trust_remote_code``.
* Loads in full BF16 (~52 GB of weights), so it needs a big GPU -- an 80 GB A100 /
  H100 holds it with room for activations.
* Quantization was removed on purpose: bitsandbytes only quantizes the
  attention/router ``nn.Linear`` layers and leaves the batched MoE expert tensors
  (``DiffusionGemmaTextExperts``, the bulk of the 26B params) in BF16, so 4-bit/8-bit
  barely shrank the model (~52 GB either way) -- it just added CPU offload and
  slowed everything down. See git history if you want to revive it.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    dtype: str | torch.dtype = "auto",
    device_map: str = "auto",
    hf_token: str | None = None,
    eval_mode: bool = True,
):
    """Load DiffusionGemma and its processor in full (unquantized) precision.

    Parameters
    ----------
    model_id:
        HuggingFace repo id. Defaults to ``google/diffusiongemma-26B-A4B-it``.
    dtype:
        Load/compute dtype. ``"auto"`` (default) resolves to the checkpoint's
        native dtype (BF16). ~52 GB on the GPU either way.
    device_map:
        Passed to ``from_pretrained``. ``"auto"`` shards across available devices
        (and would spill to CPU/disk if the model doesn't fit -- so give it an
        80 GB GPU to keep everything resident).
    hf_token:
        HuggingFace token. Falls back to the ``HF_TOKEN`` env var, then to None
        (the model is public, so a token is optional).
    eval_mode:
        If True (default), put the model in eval mode.

    Returns
    -------
    tuple[DiffusionGemmaForBlockDiffusion, AutoProcessor]
        The loaded ``(model, processor)``.
    """
    # Import here so a too-old `transformers` produces an actionable error rather
    # than failing at module-import time.
    try:
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    except ImportError as exc:  # pragma: no cover - depends on installed version
        raise ImportError(
            "Could not import `DiffusionGemmaForBlockDiffusion` from `transformers`. "
            "Your transformers is likely too old for DiffusionGemma; upgrade with "
            "`pip install -U transformers`."
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN")

    model_kwargs: dict = {
        "device_map": device_map,
        "token": token,
        "dtype": dtype,
    }

    logger.info("Loading %s (BF16, device_map=%s)...", model_id, device_map)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, token=token)

    if eval_mode:
        model.eval()

    if getattr(model, "device", None) is not None and model.device.type == "cpu":
        logger.warning(
            "Model is on CPU - DiffusionGemma needs a CUDA GPU; generation will be "
            "extremely slow or fail."
        )

    return model, processor


if __name__ == "__main__":
    # Smoke test: load the model and run a single prompt through the documented
    # chat-template -> generate -> decode path.
    logging.basicConfig(level=logging.INFO)

    model, processor = load_model()

    message = [{"role": "user", "content": "Why is the sky blue?"}]
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=64)
    print(processor.decode(output[0], skip_special_tokens=True))
