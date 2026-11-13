#!/usr/bin/env python3
"""Rung 8 harvest: a rank-k refusal subspace per layer, from paired activations.

Why this exists
---------------
Everything this campaign ran projected ONE direction per layer, 24 times. The
public build that reports KL 0.0374 at 5/100 refusals has an edit whose deltas
are rank 2.3-11.3 (effective rank) at 4-14% rel_change, touching only
`o_proj` and `w2` in layers 12-19 and 22 (see `.forensics/ara/diff-vs-base.json`).
Same dose family as mine, structurally different object: a subspace, not a ray.

`_collect_paired_output_phase` gives, per layer, one mean refusal vector and one
mean compliance vector per prompt — the mean destroys every difference but the
first. This script keeps the per-prompt pairs and asks how many directions the
refusal/compliance split actually occupies:

    D[p] = h_refusal[p] - h_compliance[p]        shape (N, hidden)
    D = U S V^T                                  -> V[:, :k] = the subspace

Two deliberate choices, both recorded rather than assumed:
  * `--orthogonalize-good` projects each subspace direction off the mean
    compliance direction (Heretic's `orthogonalize_direction`), so the edit
    does not subtract "harmless-style" content.
  * rank selection reports the full singular spectrum, so k is chosen from a
    number and not from taste.

Outputs a .pt with per-layer (k, hidden) orthonormal subspaces plus a JSON with
the spectrum, separation per rank and the cosine against the rank-1 paired
direction the six previous rungs used.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/lfm3-1.7b.yaml")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--rank", type=int, default=8, help="max subspace rank to extract")
    ap.add_argument("--pair-tokens", type=int, default=48,
                    help="response tokens averaged per condition (shorter = cheaper, still post-decision)")
    ap.add_argument("--out", default=".subspace/r8-subspace.pt")
    ap.add_argument("--json", default=".subspace/r8-subspace.json")
    ap.add_argument("--orthogonalize-good", action="store_true", default=True)
    ap.add_argument("--no-orthogonalize-good", dest="orthogonalize_good", action="store_false")
    ap.add_argument("--layers", default="10,11,12,13,14,15,16,17,18,19,20,21,22,23")
    args = ap.parse_args()

    from config import load_config
    from harness.abl import _load_model_tok
    from probe import _collect_paired_output_phase, _find_layers
    from prompt_format import format_prompt
    from prompts import DEFAULT_HARMFUL

    cfg = load_config(args.config)
    model, tok = _load_model_tok(cfg)
    model.eval()
    device = next(model.parameters()).device
    layers = _find_layers(model, cfg.model_arch)
    n_layers = len(layers)
    prefill = getattr(cfg, "paired_prefill", None) or "Sure, here is"

    prompts = [format_prompt(tok, p, cfg.prompt_flavor) for p in list(DEFAULT_HARMFUL)[: args.n_prompts]]
    print(f"harvest: {len(prompts)} paired prompts, layers 0..{n_layers-1}, "
          f"{args.pair_tokens} response tokens per condition")

    refusal, affirm = _collect_paired_output_phase(
        model, tok, prompts, layers, n_layers, device, prefill, args.pair_tokens)

    want = [int(x) for x in args.layers.split(",") if int(x) < n_layers]
    out: dict[int, torch.Tensor] = {}
    table = []
    for li in want:
        if li not in refusal or li not in affirm:
            print(f"L{li}: no activations, skip"); continue
        R = torch.stack([v.float() for v in refusal[li]])   # (N, hidden)
        A = torch.stack([v.float() for v in affirm[li]])
        n = min(len(R), len(A))
        R, A = R[:n], A[:n]
        D = R - A                                            # paired differences
        D = D - D.mean(dim=0, keepdim=True)                  # centre: keep only what varies with the pair
        # rank-r subspace of the paired differences
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        k = min(args.rank, Vh.shape[0])
        V = Vh[:k]                                           # (k, hidden)

        if args.orthogonalize_good:
            g = A.mean(dim=0)
            g = g / g.norm().clamp(min=1e-8)
            V = V - (V @ g).unsqueeze(1) * g.unsqueeze(0)    # project off the good direction
            # re-orthonormalize (Gram-Schmidt, numerically stable enough at k<=16)
            q, _ = torch.linalg.qr(V.T)
            V = q.T

        # separation per rank: how far apart are the projections of the two sets
        proj_R, proj_A = R @ V.T, A @ V.T
        sep_rank = []
        for j in range(V.shape[0]):
            a, b = proj_R[:, j], proj_A[:, j]
            pooled = ((a.std(unbiased=True) ** 2 + b.std(unbiased=True) ** 2) / 2) ** 0.5
            sep_rank.append(float((a.mean() - b.mean()).abs() / pooled.clamp(min=1e-8)))

        # cosine of the top subspace direction against the rank-1 mean-difference direction
        d1 = R.mean(dim=0) - A.mean(dim=0)
        d1 = d1 / d1.norm().clamp(min=1e-8)
        cos1 = float((V[0] @ d1).abs())

        out[li] = V.to(torch.float32)
        table.append({
            "layer": li,
            "singular_values": [round(float(x), 3) for x in S[: min(6, len(S))]],
            "sv_ratio_s2_over_s1": round(float(S[1] / S[0]), 4) if len(S) > 1 else None,
            "cum_energy_top1": round(float((S[0] ** 2) / (S ** 2).sum()), 4),
            "cum_energy_top4": round(float((S[:4] ** 2).sum() / (S ** 2).sum()), 4),
            "separation_by_rank": [round(x, 3) for x in sep_rank],
            "cos_subspace1_vs_rank1_paired": round(cos1, 4),
        })
        print(f"L{li:2d} s={[round(float(x),2) for x in S[:5]]} top1_energy={table[-1]['cum_energy_top1']:.3f} "
              f"top4={table[-1]['cum_energy_top4']:.3f} sep_r1={sep_rank[0]:.2f} sep_r4={sep_rank[min(3,len(sep_rank)-1)]:.2f} "
              f"cos(v1,d1)={cos1:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    Path(args.json).write_text(json.dumps(table, indent=1))

    if table:
        e1 = statistics.mean(t["cum_energy_top1"] for t in table)
        e4 = statistics.mean(t["cum_energy_top4"] for t in table)
        mr = statistics.mean(t["sv_ratio_s2_over_s1"] for t in table if t["sv_ratio_s2_over_s1"])
        print(f"\nmean top1 energy {e1:.3f}, mean top4 {e4:.3f}, mean s2/s1 {mr:.3f}")
        if e1 > 0.9:
            print("VERDICT: the paired refusal difference is essentially rank-1 at these layers; "
                  "a subspace edit has little extra to remove and ARA's advantage must come "
                  "from somewhere else (layer subset, row-normalisation, or its optimiser).")
        else:
            print(f"VERDICT: rank-1 holds only {e1*100:.0f}% of the paired-difference energy here; "
                  "there is real subspace structure to ablate.")
    print(f"wrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
