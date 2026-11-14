#!/usr/bin/env python3
"""Reproduce a published LFM3 abliteration from its disclosed parameters, then
judge it with THIS project's instruments.

Provenance of the recipe (not invented here):
  MihaiPopa-1/LFM3-1.7B-Heretic-Max, produced by Heretic v1.4.0 (p-e-w), whose
  abliteration modifier is read from the package source (heretic 2.1.0 wheel,
  modifiers/abliteration.py):
    - residuals: generate ONE token, take the hidden state at that generated
      position, per prompt, per layer (model.py:get_residuals)
    - direction: mean(bad) - mean(good), normalised, then orthogonalised
      against the normalised mean(good) direction (orthogonalize_direction=True)
    - one global direction for all modules, chosen by a fractional layer index
      and built by lerp between the two neighbouring layer directions
    - edit: delta W = -lambda * v (v^T W) with row-magnitude restoration
      (row_normalization="full") -- a rank-1 projection, same family as MPOA
    - lambda per component follows a tent: max_weight at max_weight_position,
      decaying to min_weight at min_weight_distance

  Disclosed trial for LFM3-1.7B (their card):
    direction_index 11.53
    attn.o_proj   min 1.19  max 1.49  peak 14.92  reach 13.28
    mlp.down_proj min 1.09  max 1.23  peak 17.78  reach 11.61
    their reported: KL 0.0040, refusals 6/100 (base 93/100)

Stages
  dir   harvest good/bad residuals at the first generated token, build the
        direction, report separation + cosine against our paired direction.
  edit  apply the disclosed lambda profile to o_proj + w2 on the layer window,
        MPOA-style row-norm preservation, and save the checkpoint.
  Imports nothing from the campaign's earlier rungs except the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def tent(layer: int, max_w: float, max_pos: float, min_w: float, reach: float) -> float:
    """Heretic's WeightDistribution: linear from max_weight at max_weight_position
    down to min_weight at min_weight_distance, clamped outside that window."""
    d = abs(layer - max_pos)
    if d >= reach:
        return min_w
    return max_w + (min_w - max_w) * (d / reach)


def load_prompts(n_each: int) -> tuple[list[str], list[str]]:
    from datasets import load_dataset
    good = load_dataset("mlabonne/harmless_alpaca", split=f"train[:{n_each}]")
    bad = load_dataset("mlabonne/harmful_behaviors", split=f"train[:{n_each}]")
    g = [r["text"] for r in good]
    b = [r["text"] for r in bad]
    return g, b


def harvest_residuals(model, tok, prompts, device, n_layers: int, batch: int = 8):
    """Hidden state at the first GENERATED token, per layer, mean over batch.
    Mirrors heretic's get_residuals: generate 1 token with output_hidden_states."""
    sums = None
    count = 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False,
                                 output_hidden_states=True, return_dict_in_generate=True,
                                 use_cache=False,
                                 pad_token_id=getattr(tok, "eos_token_id", None))
        hs = out.hidden_states[0]        # first generated token -> tuple over layers
        cur = torch.stack([h[:, -1, :].float().mean(dim=0) for h in hs])  # (L+1, hidden)
        sums = cur if sums is None else sums + cur
        count += 1
    return sums / max(count, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dir", "edit", "both"])
    ap.add_argument("--config", default="models/lfm3-1.7b.yaml")
    ap.add_argument("--n-each", type=int, default=400)
    ap.add_argument("--direction-index", type=float, default=11.53)
    ap.add_argument("--out-dir", default=".r8")
    ap.add_argument("--patch", default="campaigns/liquidai-lfm3-1.7b/r8-heretic-repro")
    args = ap.parse_args()

    from config import load_config
    from harness.abl import _load_model_tok
    from probe import _find_layers

    cfg = load_config(args.config)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, tok = _load_model_tok(cfg)
    model.eval()
    device = next(model.parameters()).device
    layers = _find_layers(model, cfg.model_arch)
    n_layers = len(layers)
    print(f"loaded {cfg.model_id}: {n_layers} layers, device {device}")

    # ---------------- stage: direction ----------------
    if args.stage in ("dir", "both"):
        good, bad = load_prompts(args.n_each)
        print(f"prompts: {len(good)} good / {len(bad)} bad")
        mg = harvest_residuals(model, tok, good, device, n_layers)
        mb = harvest_residuals(model, tok, bad, device, n_layers)
        torch.save({"good": mg, "bad": mb}, out / "residual-means.pt")

        D = mb - mg                                   # (L+1, hidden)
        Dn = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G = mg / mg.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # orthogonalise each layer's direction against the good direction
        Do = Dn - (Dn * G).sum(dim=1, keepdim=True) * G
        Do = Do / Do.norm(dim=1, keepdim=True).clamp(min=1e-8)

        idx = args.direction_index
        lo, hi = int(idx), min(int(idx) + 1, n_layers)
        frac = idx - lo
        v_global = torch.nn.functional.normalize(
            Do[lo].lerp(Do[hi], frac), dim=0)          # hidden-state index == layer index

        # separation of the two prompt groups along v_global, per layer index 1..L
        sep = []
        for li in range(1, n_layers + 1):
            a = (mb[li] - mg[li]) @ v_global
            sep.append(round(float(a), 3))

        torch.save({"v": v_global, "D": Do, "idx": idx}, out / "heretic-direction.pt")

        # cosine against the campaign's own paired direction, if present
        cmp_tbl = []
        pp = Path("campaigns/liquidai-lfm3-1.7b/directions-chat-paired-t320.pt")
        if pp.exists():
            blob_ours = torch.load(pp, map_location="cpu", weights_only=False)
            # the direction file is {"dirs": {...}, "scores": ..., "model_id": ...};
            # tolerate a bare layer->tensor map too
            ours = blob_ours.get("dirs", blob_ours) if isinstance(blob_ours, dict) else blob_ours
            for li in sorted(ours, key=lambda k: int(k)):
                d = ours[li]
                d = d.reshape(-1, d.shape[-1])[0].float()
                d = d / d.norm().clamp(min=1e-8)
                cmp_tbl.append({"layer": int(li),
                                "cos_with_heretic_global": round(float((d @ v_global).abs()), 4)})
            cos_mean = sum(c["cos_with_heretic_global"] for c in cmp_tbl) / len(cmp_tbl)
            print(f"cos(our paired per-layer, heretic global): mean {cos_mean:.3f} "
                  f"min {min(c['cos_with_heretic_global'] for c in cmp_tbl):.3f} "
                  f"max {max(c['cos_with_heretic_global'] for c in cmp_tbl):.3f}")

        (out / "dir-report.json").write_text(json.dumps({
            "n_good": len(good), "n_bad": len(bad), "direction_index": idx,
            "window": [lo, hi], "frac": round(frac, 3),
            "projection_separation_by_layer": sep,
            "cos_vs_paired": cmp_tbl,
        }, indent=1))
        print("projection separation (bad-good, per layer):", sep)

    # ---------------- stage: edit ----------------
    if args.stage in ("edit", "both"):
        blob = torch.load(out / "heretic-direction.pt", map_location="cpu", weights_only=False)
        v = blob["v"].to(device)
        v = v / v.norm().clamp(min=1e-8)

        profile = {
            "o_proj": dict(max_w=1.49, max_pos=14.92, min_w=1.19, reach=13.28),
            "w2":     dict(max_w=1.23, max_pos=17.78, min_w=1.09, reach=11.61),
        }
        lo, hi = 11, 23          # the layer window the sweep implies; both tents are ~0 outside it
        from sweep import _iter_resolved_projs
        touched = []
        for li in range(lo, hi + 1):
            layer = layers[li]
            for wname, p in profile.items():
                lam = tent(li, p["max_w"], p["max_pos"], p["min_w"], p["reach"])
                for _mod, W in _iter_resolved_projs(layer, wname):
                    W0 = W.clone()
                    row0 = W.norm(dim=1, keepdim=True)
                    # delta = -lambda * v (v^T W); v lives in the residual (output) basis
                    W.sub_(lam * torch.einsum("i,j->ij", v.to(W.dtype), v.to(W.dtype) @ W))
                    # row-magnitude restoration (Heretic row_normalization="full", our MPOA)
                    row1 = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    W.mul_(row0 / row1)
                    rel = float((W0 - W).norm().item() / max(float(W0.norm().item()), 1e-12))
                    touched.append({"layer": li, "weight": wname, "alpha": round(lam, 4),
                                    "rel_change": round(rel, 5)})
                    print(f"L{li:2d} {wname:7s} alpha={lam:.3f} rel={rel*100:.2f}%")

        p = Path(args.patch); p.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(p, safe_serialization=True)
        tok.save_pretrained(p)
        (p.parent / "r8-edit-report.json").write_text(json.dumps(touched, indent=1))
        print(f"saved checkpoint -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
