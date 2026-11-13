#!/usr/bin/env python3
"""Diff a public 'abliterated' LFM3 build against its base, in weight space.

Purpose: the campaign established that its own rank-1 direction removes refusal
only by damaging the model. Public builds report KL 0.004-0.037 with single-digit
refusal rates. Either they are doing something structurally different, or their
refusal measurement is blind. This script answers the weight-space half.

For every tensor that differs it reports:
  rel_change   |dW| / |W|            — how much of the matrix was touched
  rank1_frac   s1 / sum(s)           — how much of dW is one direction
  top5         the five largest singular values, normalised by s1
  row_norms    max relative change of a row norm (MPOA-style edits preserve these)

A rank-1 projection at any alpha gives rank1_frac ~= 1.0 and top5 ~= [1,0,0,0,0].
An L-BFGS / ARA edit is expected to be higher rank and spread across rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_shards(path: Path) -> dict[str, torch.Tensor]:
    """Load a single safetensors file or a sharded index directory."""
    from safetensors.torch import load_file
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        want = set(json.loads(idx.read_text())["weight_map"].values())
        out: dict[str, torch.Tensor] = {}
        for shard in sorted(want):
            out.update(load_file(path / shard, device="cpu"))
        return out
    if (path / "model.safetensors").exists():
        return load_file(path / "model.safetensors", device="cpu")
    raise SystemExit(f"no safetensors found under {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--other", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="only print the N largest")
    args = ap.parse_args()

    base = load_shards(args.base)
    other = load_shards(args.other)
    shared = [k for k in base if k in other and base[k].shape == other[k].shape]
    print(f"base tensors {len(base)}, other {len(other)}, comparable {len(shared)}")

    rows = []
    for k in shared:
        a, b = base[k].float(), other[k].float()
        if a.dim() not in (1, 2):
            continue
        d = (b - a)
        nW = a.norm().item()
        nD = d.norm().item()
        if nD == 0 or nW == 0:
            continue
        rel = nD / nW
        if rel < 1e-6:
            continue
        # rank profile of the delta (2-D only; 1-D = bias/scale tensors)
        if d.dim() == 2 and min(d.shape) > 1:
            s = torch.linalg.svdvals(d)
            r1 = float(s[0] / s.sum())
            top5 = [round(float(x / s[0]), 3) for x in s[:5]]
            eff_rank = float(s.sum() ** 2 / (s ** 2).sum())
        else:
            r1, top5, eff_rank = 1.0, [1.0], 1.0
        # row-norm preservation
        if d.dim() == 2 and a.dim() == 2:
            rn_a = a.norm(dim=1)
            rn_b = b.norm(dim=1)
            rn = float(((rn_b - rn_a).abs() / rn_a.clamp(min=1e-8)).max())
        else:
            rn = 0.0
        rows.append({"tensor": k, "shape": list(a.shape), "rel_change": rel,
                     "rank1_frac": r1, "eff_rank": eff_rank, "top5": top5,
                     "row_norm_max_rel": rn})

    rows.sort(key=lambda r: -r["rel_change"])
    if args.limit:
        rows = rows[: args.limit]
    print(f"changed tensors: {len(rows)}\n")
    print(f"{'tensor':44s} {'shape':14s} {'rel_chg':>8s} {'rank1':>6s} {'effrk':>6s} {'rownorm':>8s} top5")
    for r in rows:
        print(f"{r['tensor'][:44]:44s} {str(r['shape']):14s} {r['rel_change']*100:7.3f}% "
              f"{r['rank1_frac']:6.3f} {r['eff_rank']:6.2f} {r['row_norm_max_rel']*100:7.3f}% {r['top5']}")

    out = args.other / "diff-vs-base.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
