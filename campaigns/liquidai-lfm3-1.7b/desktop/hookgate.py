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


def _encode(tok, prompt: str, device: str, thinking: bool = True):
    # The template's DEFAULT renders a bare "<think>" (thinking ON). Every arm in
    # rungs 10-15 ran in that regime; both popular public builds score with it OFF,
    # which renders "<think></think>". Never leave this implicit again.
    kw = {} if thinking else {"enable_thinking": False}
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", **kw)
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
def generate(model, tok, prompts, device, max_new: int, dirs=None,
             strength: float = 1.0, where: str = "post", prefill_only: bool = False,
             layers: set[int] | None = None, thinking: bool = True):
    """where='post' projects the norm's OUTPUT (y-space, the rung-10 endpoint).

    where='pre' projects the branch's raw write BEFORE the norm, along
    a = gamma * r. That is the operator a weight-space abliteration implements:
    a linear map on x whose null space contains a. The two are not the same
    operator once gamma is far from 1, which is exactly what F1 kept saying.

    prefill_only=True edits the prompt pass and leaves the generation passes
    untouched. A decode step presents seq_len 1; a prompt pass presents seq_len
    prompt_len, so the shape is the discriminator. rSalone's own card reports
    prompt-side directions beating thinking-side ones, which is this idea stated
    from their side of the work.
    """
    tally = {"prompt_passes": 0, "decode_passes": 0, "edited": 0}
    hooks = []
    if dirs:
        for li, layer in enumerate(model.model.layers):
            for place, norm_suffix in PLACES.items():
                key = f"L{li}.{place}"
                if key not in dirs:
                    continue
                if layers is not None and li not in layers:
                    continue
                r = dirs[key].to(device=device, dtype=torch.bfloat16)
                mod = getattr(layer, norm_suffix, None)
                if mod is None:
                    continue

                if where == "pre":
                    g = mod.weight.detach().to(device=device, dtype=torch.float32)
                    a = (g * r.float())
                    a = (a / a.norm().clamp(min=1e-8)).to(dtype=torch.bfloat16)

                    def _p(_m, args, a=a):
                        x = args[0]
                        if x.shape[1] == 1:
                            tally["decode_passes"] += 1
                        else:
                            tally["prompt_passes"] += 1
                        if prefill_only and x.shape[1] == 1:
                            return None
                        tally["edited"] += 1
                        return (x - strength * (x @ a).unsqueeze(-1) * a,) + tuple(args[1:])

                    hooks.append(mod.register_forward_pre_hook(_p))
                else:
                    def _h(_m, _i, out, r=r):
                        return out - strength * (out @ r).unsqueeze(-1) * r

                    hooks.append(mod.register_forward_hook(_h))
    out = []
    try:
        for p in prompts:
            ids = _encode(tok, p, device, thinking=thinking)
            gen = model.generate(input_ids=ids, max_new_tokens=max_new,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True)
            out.append(text)
    finally:
        for h in hooks:
            h.remove()
    generate.tally = tally
    print(f"  hook tally: prompt_passes {tally['prompt_passes']} "
          f"decode_passes {tally['decode_passes']} edited {tally['edited']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/lfm3-1.7b-r8.yaml")
    ap.add_argument("--dirs", default="/out/r9/r9-directions.pt")
    ap.add_argument("--position", default="mean", choices=["mean", "last"])
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--where", default="post", choices=["post", "pre"])
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke tests)")
    ap.add_argument("--layers", default="", help="layer window, e.g. '0-11' or '12,13,14'")
    ap.add_argument("--out", default="/out/r10/hookgate.json")
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="", choices=["", "hook", "base"],
                    help="generate a single arm; default both")
    ap.add_argument("--model-id", default=MODEL_ID,
                    help="load a different checkpoint, e.g. a public abliterated build")
    ap.add_argument("--no-think", action="store_true",
                    help="render <think></think> (thinking disabled), the regime public cards report in")
    ap.add_argument("--base-from", default="",
                    help="reuse the base arm from an earlier hookgate.json")
    a = ap.parse_args()

    from config import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(a.config)
    max_new = a.max_new or int(getattr(cfg, "gate_refusal_max_new_tokens", 1024) or 1024)
    layers = None
    if a.layers:
        layers = set()
        for part in a.layers.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                layers.update(range(int(lo), int(hi) + 1))
            else:
                layers.add(int(part))
        print(f"layer window {sorted(layers)}")

    prompts = _held_out(cfg)
    if a.limit:
        prompts = prompts[:a.limit]
    thinking = not a.no_think
    print(f"held-out prompts {len(prompts)} | window {max_new} | strength {a.strength} | where {a.where}"
          f" | prefill_only {a.prefill_only} | model {a.model_id} | thinking {thinking}", flush=True)

    blob = torch.load(a.dirs, map_location="cpu", weights_only=False)
    dirs = blob["dirs"][a.position]
    print(f"directions: {len(dirs)} places from {a.dirs} ({a.position})")

    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()

    tail = tok.apply_chat_template([{"role": "user", "content": "probe"}],
                                   add_generation_prompt=True, tokenize=False,
                                   **( {} if thinking else {"enable_thinking": False} ))[-24:]
    print(f"template tail {tail!r}  (thinking={thinking})", flush=True)
    assert ("</think>" in tail) == (not thinking), "template regime does not match --no-think"

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
                         dirs=None if arm == "base" else dirs, strength=a.strength,
                         where=a.where, prefill_only=a.prefill_only, layers=layers,
                         thinking=thinking)
        for p, t in zip(prompts, texts):
            gens.append({"prompt": p, "response": t, "arm": arm, "gate": "refusal"})
        print(f"  {arm}: {len(texts)} generations, mean chars {sum(len(t) for t in texts)//max(len(texts),1)}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": a.config, "position": a.position,
                               "strength": a.strength, "window": max_new,
                               "where": a.where, "prefill_only": a.prefill_only,
                               "layers": a.layers, "model_id": a.model_id,
                               "thinking": thinking,
                               "hook_tally": getattr(generate, "tally", None),
                               "generations": gens}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
