#!/usr/bin/env python
"""Cross-layer cosine of top singular vectors of (other - base) per tensor.

The question this answers: is a published edit one direction applied at every
layer, or a different direction per layer? Both look identical in a rel_change
table. A cosine matrix of the leading left singular vector of each tensor's
delta separates them.

Usage:
  python cross_layer_cosine.py --base .forensics/base --other .forensics/heretic \
      [--component w2] [--out .forensics/heretic/cross-cos.json]

Reports per component: mean |cos| over layer pairs, mean |cos| between adjacent
layers, and the participation ratio of the stacked deltas (effective number of
independent directions).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.weight$")


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    f = path / "model.safetensors" if path.is_dir() else path
    return load_file(str(f), device="cpu")


def _index(sd: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = _LAYER_RE.search(k)
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--component", default=None,
                    help="restrict to one component (e.g. w2); default = all changed")
    ap.add_argument("--min-rel-change", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = _index(_load_safetensors(Path(a.base)))
    other = _index(_load_safetensors(Path(a.other)))
    layers = sorted(set(base) & set(other))

    report: dict = {"base": str(a.base), "other": str(a.other), "components": {}}
    comps = set()
    for li in layers:
        for comp in set(base[li]) & set(other[li]):
            comps.add(comp)

    for comp in sorted(comps):
        if a.component and comp != a.component:
            continue
        vecs: dict[int, torch.Tensor] = {}
        rels: dict[int, float] = {}
        for li in layers:
            if comp not in base[li] or comp not in other[li]:
                continue
            W0 = base[li][comp].float()
            W1 = other[li][comp].float()
            if W0.shape != W1.shape:
                continue
            dW = W1 - W0
            rel = float(dW.norm() / W0.norm().clamp(min=1e-12))
            if rel < a.min_rel_change:
                continue
            # leading left singular vector: which output directions moved
            u, _, _ = torch.linalg.svd(dW, full_matrices=False)
            vecs[li] = u[:, 0]
            rels[li] = round(rel, 6)
        if len(vecs) < 2:
            continue
        ls = sorted(vecs)
        M = torch.stack([vecs[li] for li in ls])            # (n_layers, hidden)
        C = (M @ M.T).abs()
        n = len(ls)
        off = C[~torch.eye(n, dtype=torch.bool)]
        adj = torch.tensor([float(C[i, i + 1]) for i in range(n - 1)])
        # participation ratio of the singular-value spectrum of the stacked deltas
        s = torch.linalg.svdvals(M)
        pr = float((s.sum() ** 2) / (s ** 2).sum().clamp(min=1e-12))
        report["components"][comp] = {
            "n_layers": n,
            "layers": ls,
            "rel_change": rels,
            "cos_mean": round(float(off.mean()), 4),
            "cos_adjacent_mean": round(float(adj.mean()), 4) if len(adj) else None,
            "cos_min": round(float(off.min()), 4),
            "cos_max": round(float(off.max()), 4),
            "spectrum_participation_ratio": round(pr, 3),
            "matrix": [[round(float(x), 3) for x in row] for row in C],
        }

    print(f"{'component':22s} {'layers':>6s} {'cos_mean':>9s} {'adj':>6s} "
          f"{'eff_dirs':>9s}  rel_change range")
    for comp, r in report["components"].items():
        rv = list(r["rel_change"].values())
        print(f"{comp:22s} {r['n_layers']:6d} {r['cos_mean']:9.3f} "
              f"{r['cos_adjacent_mean'] if r['cos_adjacent_mean'] is not None else float('nan'):6.3f} "
              f"{r['spectrum_participation_ratio']:9.2f}  {min(rv):.4f}-{max(rv):.4f}")

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
