#!/usr/bin/env python
"""Recover a hook-time direction from somebody else's baked weights.

A public abliterated build is a set of weight deltas. For every place that writes
into the residual stream we take the top left singular vector of (other - base).
That vector is the direction the build removed from the row space, so projecting it
out of the post-norm output at generation time reproduces their edit as behaviour,
without needing their algebra to be faithful on this checkpoint.

Sign does not matter: removing a one-dimensional subspace is the same operation for
u and -u.

Emits the same layout `hookgate.py` reads: {"dirs": {"mean": {...}, "last": {...}}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

SUFFIX_TO_PLACE = {
    ".self_attn.o_proj.weight": "operator",
    ".conv.out_proj.weight": "operator",
    ".feed_forward.w2.weight": "ffn",
}


def _place(name: str):
    for suf, place in SUFFIX_TO_PLACE.items():
        if name.endswith(suf):
            layer = name.split(".")[2]
            return f"L{layer}.{place}"
    return None


def _top_left(dW: torch.Tensor, iters: int = 12, seed: int = 20261114) -> torch.Tensor:
    """Top left singular vector of dW by power iteration on dW dW^T."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dW.shape[0], generator=g, dtype=torch.float32)
    v = v / v.norm().clamp(min=1e-12)
    for _ in range(iters):
        v = dW @ (dW.T @ v)
        v = v / v.norm().clamp(min=1e-12)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    B = str(Path(a.base) / "model.safetensors")
    O = str(Path(a.other) / "model.safetensors")
    dirs, rows = {}, []
    with safe_open(O, framework="pt") as o, safe_open(B, framework="pt") as b:
        names = [n for n in o.keys() if _place(n) and n in b.keys()]
        for name in sorted(names, key=lambda n: (int(n.split(".")[2]), n)):
            w0 = b.get_tensor(name).float()
            dW = o.get_tensor(name).float() - w0
            if float(dW.norm()) < 1e-9:
                continue
            u = _top_left(dW)
            u = u / u.norm().clamp(min=1e-8)
            dirs[_place(name)] = u
            rel = float(dW.norm() / w0.norm().clamp(min=1e-8))
            rows.append((_place(name), rel, float(dW.norm())))
            del dW, w0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dirs": {"mean": dirs, "last": dirs},
                "model_id": a.other, "source": "weight-delta u1 (power iteration)"}, out)
    print(f"{len(dirs)} places from {a.other}")
    print(f"{'place':14s} rel_change  dW_norm")
    for place, rel, n in sorted(rows, key=lambda r: -r[1])[:12]:
        print(f"{place:14s} {rel:9.4%}  {n:8.3f}")
    (out.with_suffix(".json")).write_text(json.dumps(
        {p: {"rel_change": r, "dW_norm": n} for p, r, n in rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
