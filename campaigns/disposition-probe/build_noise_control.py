#!/usr/bin/env python3
"""Build the attribution control: pristine weights + matched-magnitude noise.

The abliterated build differs from pristine at rel_l2 0.022-0.030 per tensor
(campaigns/lfm2.5-all32-replication). This writes a third arm whose weights are
perturbed by Gaussian noise of the SAME per-tensor relative L2 norm, so any
disposition shift in that arm is attributable to perturbation magnitude alone,
not to the removal of a direction.

Usage:
  python build_noise_control.py --model LiquidAI/LFM2.5-1.2B-Instruct \
      --out campaigns/disposition-probe/models/noise-0.025 --rel 0.025
"""
from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20261113


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rel", type=float, default=0.025)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)

    touched = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() < 2:
                continue
            n = torch.randn_like(p, dtype=torch.float32)
            scale = args.rel * p.float().norm() / n.norm()
            p.add_((scale * n).to(p.dtype))
            touched += 1
    print(f"perturbed {touched} tensors at rel_l2 {args.rel}")
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
