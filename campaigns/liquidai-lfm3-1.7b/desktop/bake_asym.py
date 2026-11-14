#!/usr/bin/env python
"""Rung 9 — bake an LFM3 refusal ablation that respects the sandwich norm.

Why this exists
---------------
Every residual write in LFM3-1.7B is followed by its own RMSNorm with learned
gain gamma:

    hidden = post_operator_norm(attn_or_conv(operator_norm(h))) + h
    hidden = hidden + post_ffn_norm(feed_forward(ffn_norm(hidden)))

so the refusal direction r is measured *after* a gamma. Baking the symmetric
rank-1 edit `W -= lam v v^T W` (what `excise.py` does) removes a component of the
pre-norm write, not the measured direction. rSalone/LFM3-1.7B-uncensored-abl48
measured that mismatch on this model: symmetric baking gives KL 0.0053 against
generation-time removal, asymmetric baking gives 0.0013.

The correct baked edit pairs a = gamma*r with b = r/gamma, scaled so a.b = 1:

    W' = W - lam * b (a^T W)

Subcommands
-----------
  dir     harvest r at all 48 places (24 layers x {operator, ffn}), prompt-side
  bake    write the asymmetric edit into the weights, save a checkpoint
  check   faithfulness: KL(baked vs hooked removal) against KL(baked vs base)

Nothing here writes outside --out-dir / --patch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MODEL_ID = "LiquidAI/LFM3-1.7B"
N_LAYERS = 24
# place name -> (norm module suffix, projection attribute path)
PLACES = {
    "operator": ("post_operator_norm", "self_attn.o_proj"),   # conv layers use conv.out_proj
    "ffn": ("post_ffn_norm", "feed_forward.w2"),
}


# ---------------------------------------------------------------- prompts
def load_prompts(n: int) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    def col(ds, want):
        for k in want:
            if k in ds.column_names:
                return [str(x) for x in ds[k] if str(x).strip()]
        raise SystemExit(f"no usable column in {ds.column_names}")

    bad = load_dataset("mlabonne/harmful_behaviors", split="train")
    good = load_dataset("mlabonne/harmless_alpaca", split="train")
    bad_p = col(bad, ["prompt", "instruction", "text"])[:n]
    good_p = col(good, ["prompt", "instruction", "text"])[:n]
    return bad_p, good_p


def _encode(tok, prompt: str, device: str):
    """Chat-template the prompt. transformers 5.x returns a BatchEncoding."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids.to(device)


# ---------------------------------------------------------------- harvest
@torch.no_grad()
def harvest(model, tok, prompts: list[str], device: str) -> dict:
    """Mean residual-branch output after the post-norm, per place.

    Returns {"last": {place: tensor}, "mean": {place: tensor}} where place is
    "L{i}.operator" / "L{i}.ffn". Only prompt positions are read: the model is
    forwarded on the templated prompt alone, so "mean" is the mean over all
    prompt tokens and "last" is the last prompt token.
    """
    cur: dict[str, torch.Tensor] = {}
    hooks = []
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        for place, (norm_suffix, _) in PLACES.items():
            mod = getattr(layer, norm_suffix, None)
            if mod is None:
                continue
            key = f"L{li}.{place}"

            def _hook(_m, _i, out, key=key):
                cur[key] = out.detach()[0].float().clone()

            hooks.append(mod.register_forward_hook(_hook))

    mean_acc: dict[str, list[torch.Tensor]] = {}
    last_acc: dict[str, list[torch.Tensor]] = {}
    try:
        for p in prompts:
            cur.clear()
            ids = _encode(tok, p, device)
            model(input_ids=ids)
            for key, h in cur.items():
                mean_acc.setdefault(key, []).append(h.mean(dim=0))
                last_acc.setdefault(key, []).append(h[-1])
    finally:
        for h in hooks:
            h.remove()

    if not mean_acc:
        raise SystemExit("harvest captured nothing")
    mean = {k: torch.stack(v).mean(dim=0) for k, v in mean_acc.items()}
    last = {k: torch.stack(v).mean(dim=0) for k, v in last_acc.items()}
    return {"last": last, "mean": mean}


def _direction(bad: torch.Tensor, good: torch.Tensor) -> torch.Tensor:
    """mean(bad) - mean(good), orthogonalised against mean(good), unit norm."""
    g = good / good.norm().clamp(min=1e-8)
    r = bad - good
    r = r - (r @ g) * g
    return r / r.norm().clamp(min=1e-8)


def cmd_dir(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32,
                                                 trust_remote_code=True).to(device).eval()
    bad_p, good_p = load_prompts(args.n_each)
    print(f"prompts: {len(bad_p)} harmful, {len(good_p)} harmless")

    B = harvest(model, tok, bad_p, device)
    G = harvest(model, tok, good_p, device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {}
    sep = {}
    gamma_stats = {}
    for pos in ("last", "mean"):
        dirs[pos] = {}
        sep[pos] = {}
        for key in B[pos]:
            if key not in G[pos]:
                continue
            r = _direction(B[pos][key], G[pos][key])
            dirs[pos][key] = r
            sep[pos][key] = round(float((B[pos][key] - G[pos][key]) @ r), 4)
        print(f"  {pos}: {len(dirs[pos])} places, "
              f"separation mean {sum(sep[pos].values())/max(len(sep[pos]),1):.3f}")

    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        for place, (norm_suffix, _) in PLACES.items():
            mod = getattr(layer, norm_suffix, None)
            if mod is None:
                continue
            w = mod.weight.detach().float().cpu()
            gamma_stats[f"L{li}.{place}"] = {
                "mean": round(float(w.mean()), 4),
                "std": round(float(w.std()), 4),
                "min": round(float(w.min()), 4),
                "max": round(float(w.max()), 4),
                "rel_dist_from_one": round(float((w - 1).norm() / w.norm()), 4),
            }

    torch.save({"dirs": dirs, "sep": sep, "model_id": MODEL_ID,
                "n_each": args.n_each, "positions": ["last", "mean"]},
               out / "r9-directions.pt")
    (out / "r9-dir-report.json").write_text(json.dumps({
        "n_each": args.n_each,
        "positions": {p: {"n_places": len(dirs[p]),
                          "separation": sep[p]} for p in dirs},
        "gamma": gamma_stats,
    }, indent=1))
    gm = [v["rel_dist_from_one"] for v in gamma_stats.values()]
    print(f"gamma distance from 1: mean {sum(gm)/len(gm):.4f} max {max(gm):.4f}"
          "  <- the number that makes the symmetric edit wrong")
    print(f"wrote {out/'r9-directions.pt'}")
    return 0


# ---------------------------------------------------------------- bake
def _proj_module(layer, place: str):
    if place == "ffn":
        return layer.feed_forward.w2, "feed_forward.w2.weight"
    if getattr(layer, "is_attention_layer", False):
        return layer.self_attn.o_proj, "self_attn.o_proj.weight"
    return layer.conv.out_proj, "conv.out_proj.weight"


def cmd_bake(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = Path(args.out_dir) / "r9-directions.pt"
    blob = torch.load(src, map_location="cpu", weights_only=False)
    dirs = blob["dirs"][args.position]
    device = args.device
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()

    rows = []
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        for place, (norm_suffix, _) in PLACES.items():
            key = f"L{li}.{place}"
            if key not in dirs:
                continue
            r = dirs[key].to(device=device, dtype=torch.bfloat16)
            proj_mod, wname = _proj_module(layer, place)
            norm = getattr(layer, norm_suffix)
            gamma = norm.weight.detach().to(device=device, dtype=torch.bfloat16)
            W = proj_mod.weight.detach()
            W0 = W.clone().float()

            a = gamma * r
            b = r / gamma.clamp(min=1e-6)
            s = (a @ b)
            if abs(float(s)) < 1e-6:
                print(f"  {key}: a.b = {float(s):.2e}; skipping")
                continue
            a = a / s                                   # so that a.b == 1
            # W' = W - lam * b (a^T W)
            upd = torch.outer(b, a @ W)
            new = (W.float() - args.strength * upd).to(W.dtype)
            proj_mod.weight.data.copy_(new)
            d = (new.float() - W0)
            rows.append({"place": key, "weight": wname,
                         "strength": args.strength,
                         "rel_change": round(float(d.norm() / W0.norm().clamp(min=1e-12)), 6),
                         "a_dot_b": round(float(s), 6)})

    rels = [r["rel_change"] for r in rows]
    print(f"baked {len(rows)} tensors, rel_change {min(rels):.4f}-{max(rels):.4f} "
          f"mean {sum(rels)/len(rels):.4f}")

    patch = Path(args.patch)
    patch.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(patch)
    tok.save_pretrained(patch)
    (patch / "r9-edit-report.json").write_text(json.dumps(rows, indent=1))
    print(f"saved {patch}")
    return 0


# ---------------------------------------------------------------- check
@torch.no_grad()
def cmd_check(args) -> int:
    """KL(baked || hooked) vs KL(baked || base) on harmless prompts.

    Hooked = remove r from the post-norm output at generation time, which is the
    thing the bake is supposed to reproduce. If the bake is faithful the first
    number is far below the second.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    blob = torch.load(Path(args.out_dir) / "r9-directions.pt", map_location="cpu",
                      weights_only=False)
    dirs = blob["dirs"][args.position]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    bad_p, good_p = load_prompts(args.n_each)
    prompts = good_p[: args.n_prompts]

    def load(p=None):
        if p:
            return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16,
                                                        trust_remote_code=True).to(device).eval()
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                    trust_remote_code=True).to(device).eval()

    def logprobs(model, hook=False):
        hooks = []
        if hook:
            for li in range(N_LAYERS):
                layer = model.model.layers[li]
                for place, (norm_suffix, _) in PLACES.items():
                    key = f"L{li}.{place}"
                    if key not in dirs:
                        continue
                    r = dirs[key].to(device=device, dtype=torch.bfloat16)
                    mod = getattr(layer, norm_suffix)

                    def _h(_m, _i, out, r=r):
                        return out - (out @ r).unsqueeze(-1) * r

                    hooks.append(mod.register_forward_hook(_h))
        out = []
        try:
            for p in prompts:
                ids = _encode(tok, p, device)
                lg = model(input_ids=ids).logits[0, -1].float().log_softmax(-1)
                out.append(lg.cpu())
        finally:
            for h in hooks:
                h.remove()
        return torch.stack(out)

    base = load()
    lp_base = logprobs(base)
    lp_hook = logprobs(base, hook=True)
    del base
    torch.cuda.empty_cache()

    baked = load(args.patch)
    lp_baked = logprobs(baked)

    def kl(p, q):
        return float((p.exp() * (p - q)).sum(-1).mean())

    k_hook = kl(lp_hook, lp_baked)
    k_base = kl(lp_base, lp_baked)
    ratio = k_hook / k_base if k_base > 0 else float("nan")
    print(f"KL(baked || hooked) {k_hook:.5f}   KL(baked || base) {k_base:.5f}   "
          f"ratio {ratio:.3f}   <- faithful bake when well below 1")
    (Path(args.out_dir) / "r9-faithfulness.json").write_text(json.dumps({
        "position": args.position, "strength": args.strength,
        "n_prompts": args.n_prompts,
        "kl_baked_vs_hooked": k_hook, "kl_baked_vs_base": k_base, "ratio": ratio,
    }, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dir", "bake", "check"])
    ap.add_argument("--out-dir", default="/out/r9")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--n-each", type=int, default=200)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--position", default="mean", choices=["mean", "last"])
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    return {"dir": cmd_dir, "bake": cmd_bake, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
