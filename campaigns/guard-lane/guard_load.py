#!/usr/bin/env python3
"""Guard-lane model loader with optional device-map / offload control.

Big guards exceed the desktop card (Qwen3Guard-Gen-4B bf16 ≈ 8.3 GB on an 8 GB
GPU): load those with device_map="auto" + a max_memory budget so accelerate
offloads the overflow to CPU instead of OOMing. Small guards (0.6B) keep the
historical device_map="cuda" behaviour unless the env vars below are set.

Env contract (both optional):
  GUARD_DEVICE_MAP   override the device_map (e.g. "auto")
  GUARD_MAX_MEMORY   max_memory budget, e.g. "0:7GiB,cpu:9GiB"
                     (implies device_map="auto" when GUARD_DEVICE_MAP is unset)

Measured on the 3060 Ti (8 GB): 4B loads in ~45 s with {0: 7GiB, cpu: 9GiB}
→ 34 modules on GPU / 6 on CPU; reads run at ~4 s/token (CPU-side attention).
"""
from __future__ import annotations

import os


def resolve_load_kwargs() -> dict:
    """Device-map kwargs for from_pretrained from the env contract above."""
    dm = os.environ.get("GUARD_DEVICE_MAP")
    mm_env = os.environ.get("GUARD_MAX_MEMORY")
    kw: dict = {}
    if mm_env:
        mm: dict = {}
        for pair in mm_env.split(","):
            key, val = pair.split(":", 1)
            mm[int(key) if key.strip().isdigit() else key.strip()] = val.strip()
        kw["max_memory"] = mm
        if dm is None:
            dm = "auto"
    kw["device_map"] = dm or "cuda"
    return kw


def load_guard_model(model_id: str, dtype, **extra):
    """from_pretrained with the env-driven device map (dtype/torch_dtype fallback)."""
    from transformers import AutoModelForCausalLM

    kwargs = resolve_load_kwargs()
    kwargs.update(extra)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    model.eval()
    return model
