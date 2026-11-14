"""How much of each write matrix's energy lies along a candidate refusal axis.

Whose question this answers. Rung 18's bake moved each write matrix's Frobenius norm by
0.03% while Heretic's published build moves its matrices by 2.5-3.1%. Removing one direction
from a 2048-wide matrix should change the norm by about 0.5*(1/2048) = 0.024% if the axis is
generic, so 0.03% is the value a *random* axis produces. A published build at 3% is removing an
axis the matrices actually write along.

The statistic is the relative energy of the write along r:

    rho = || r^T W ||_F / || W ||_F        r unit, r in residual (output) space

For a generic direction rho is about 1/sqrt(d_out) = 1/sqrt(2048) = 0.022. A build whose edit
lands at 3% norm change has rho about 0.24. That is the number to compare across directions,
because it says how much edit a given strength buys, i.e. how efficiently the axis is coupled
to the weights.

This is measurement only. It does not decide whether a direction removes refusals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

DIRS = {
    "ours-mean": (".forensics/r9/r9/r9-directions.pt", "mean"),
    "ours-last": (".forensics/r9/r9/r9-directions.pt", "last"),
    "heretic": (".forensics/r11/dirs-heretic.pt", None),
    "rsalone": (".forensics/r11/dirs-rsalone.pt", None),
}


def tensor_name(cfg: dict, li: int, place: str) -> str:
    types = cfg.get("layer_types") or []
    kind = types[li] if li < len(types) else "conv"
    if place == "ffn":
        return f"model.layers.{li}.feed_forward.w2.weight"
    if kind == "conv":
        return f"model.layers.{li}.conv.out_proj.weight"
    return f"model.layers.{li}.self_attn.o_proj.weight"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=".forensics/base")
    ap.add_argument("--out", default=".forensics/alignment.json")
    a = ap.parse_args()

    wdir = Path(a.weights)
    cfg = json.loads((wdir / "config.json").read_text())
    n_layers = int(cfg.get("num_hidden_layers", 24))

    report = {}
    with safe_open(str(wdir / "model.safetensors"), framework="pt") as f:
        keys_present = set(f.keys())
        for name, (path, pos) in DIRS.items():
            p = Path(path)
            if not p.exists():
                print(f"skip {name}: {p} missing")
                continue
            blob = torch.load(p, map_location="cpu", weights_only=False)
            dirs = blob["dirs"][pos] if pos else (blob["dirs"] if "dirs" in blob
                                                 else blob)
            if pos is None and isinstance(dirs, dict) and "mean" in dirs:
                dirs = dirs["mean"]
            rho = {}
            for li in range(n_layers):
                for place in ("operator", "ffn"):
                    key = f"L{li}.{place}"
                    if key not in dirs:
                        continue
                    tn = tensor_name(cfg, li, place)
                    if tn not in keys_present:
                        continue
                    W = f.get_tensor(tn).float()
                    r = dirs[key].float().flatten()
                    if r.shape[0] != W.shape[0]:
                        continue
                    num = (r @ W).norm().item()
                    den = W.norm().item()
                    rho[key] = round(num / max(den, 1e-9), 5)
            if not rho:
                print(f"skip {name}: no comparable tensors")
                continue
            vals = sorted(rho.values())
            report[name] = {
                "n_places": len(rho),
                "mean": round(sum(vals) / len(vals), 5),
                "median": vals[len(vals) // 2],
                "max": vals[-1],
                "generic_2048": round(2048 ** -0.5, 5),
                "implied_norm_change_pct_at_s1": round(50 * (sum(vals) / len(vals)) ** 2, 3),
                "top": sorted(rho.items(), key=lambda kv: -kv[1])[:4],
                "rho": rho,
            }
            print(f"{name:10s} n={len(rho):2d} rho mean {report[name]['mean']:.4f} "
                  f"max {report[name]['max']:.4f}  (generic {report[name]['generic_2048']})"
                  f"  top {report[name]['top'][:2]}")
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
