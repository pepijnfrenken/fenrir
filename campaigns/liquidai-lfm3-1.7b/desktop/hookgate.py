#!/usr/bin/env python
"""Rung 10 — test the direction with hooks, not with a bake.

Why: rungs 9/9b failed the faithfulness check twice. LFM3's sandwich post-norm
gains are tiny (mean 0.014), so removing a direction from the normalised branch
y = gamma * (x/rms) requires changing x by about ||x||, and RMSNorm then
re-normalises, so the linear weight bake cannot reproduce the hook edit. Baking is
an engineering problem for later; the scientific question is whether the harvested
direction removes refusal at a survivable dose, and that can be measured directly
at generation time.

Arms (same prompts, same seed, same window):
  base   no intervention
  hook   remove r at all 48 post-norm places (24 layers x {operator, ffn})

Writes a harness-shaped transcript so `.judge_transcript.py` can label it blind.
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
PLACES = {"operator": "post_operator_norm", "ffn": "post_ffn_norm"}


def _encode(tok, prompt: str, device: str):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids.to(device)


def _held_out(cfg) -> list[str]:
    from eval_split import build_split
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    h, g = list(DEFAULT_HARMFUL), list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    n_test = int(getattr(cfg, "n_test_prompts", 5) or 5)
    if n - 2 * n_test < 1:
        n_test = max(1, (n - 1) // 2)
    split = build_split(h[:n], g[:n], train_size=n - 2 * n_test, tune_size=n_test,
                        test_size=n_test, seed=cfg.eval_split_seed)
    return list(split.test)


@torch.no_grad()
def generate(model, tok, prompts, device, max_new: int, dirs=None, strength: float = 1.0):
    hooks = []
    if dirs:
        for li, layer in enumerate(model.model.layers):
            for place, norm_suffix in PLACES.items():
                key = f"L{li}.{place}"
                if key not in dirs:
                    continue
                r = dirs[key].to(device=device, dtype=torch.bfloat16)
                mod = getattr(layer, norm_suffix, None)
                if mod is None:
                    continue

                def _h(_m, _i, out, r=r):
                    return out - strength * (out @ r).unsqueeze(-1) * r

                hooks.append(mod.register_forward_hook(_h))
    out = []
    try:
        for p in prompts:
            ids = _encode(tok, p, device)
            gen = model.generate(input_ids=ids, max_new_tokens=max_new,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True)
            out.append(text)
    finally:
        for h in hooks:
            h.remove()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/lfm3-1.7b-r8.yaml")
    ap.add_argument("--dirs", default="/out/r9/r9-directions.pt")
    ap.add_argument("--position", default="mean", choices=["mean", "last"])
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", default="/out/r10/hookgate.json")
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="", choices=["", "hook", "base"],
                    help="generate a single arm; default both")
    ap.add_argument("--base-from", default="",
                    help="reuse the base arm from an earlier hookgate.json")
    a = ap.parse_args()

    from config import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(a.config)
    max_new = a.max_new or int(getattr(cfg, "gate_refusal_max_new_tokens", 1024) or 1024)
    prompts = _held_out(cfg)
    print(f"held-out prompts {len(prompts)} | window {max_new} | strength {a.strength}")

    blob = torch.load(a.dirs, map_location="cpu", weights_only=False)
    dirs = blob["dirs"][a.position]
    print(f"directions: {len(dirs)} places from {a.dirs} ({a.position})")

    dev = a.device
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()

    gens = []
    arms = (a.only,) if a.only else ("base", "hook")
    if "base" not in arms and a.base_from:
        prev = json.loads(Path(a.base_from).read_text())
        reused = [g for g in prev["generations"] if g.get("arm") == "base"]
        gens.extend(reused)
        print(f"reused {len(reused)} base generations from {a.base_from}")
    for arm in arms:
        print(f"arm {arm} ...", flush=True)
        texts = generate(model, tok, prompts, dev, max_new,
                         dirs=None if arm == "base" else dirs, strength=a.strength)
        for p, t in zip(prompts, texts):
            gens.append({"prompt": p, "response": t, "arm": arm, "gate": "refusal"})
        print(f"  {arm}: {len(texts)} generations, mean chars {sum(len(t) for t in texts)//max(len(texts),1)}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": a.config, "position": a.position,
                               "strength": a.strength, "window": max_new,
                               "generations": gens}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
