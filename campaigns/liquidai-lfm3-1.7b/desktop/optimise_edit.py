"""Optimise a per-place removal strength against a KL budget, instead of fixing it at one dose.

Why this exists. Rungs 10-17 measured one object: a single harvested direction, projected out
at ONE strength, applied identically to every place. The published builds that clear the bar
(rung 16: Heretic-Max 10 ANSWER / 2 GARBLED thinking-off) do not do that. Heretic discloses
per-layer numbers that differ from each other (attn.o_proj 1.19-1.49, peak 14.92, reach
13.28; mlp.down_proj 1.09-1.23, peak 17.78, reach 11.61), and rSalone's card describes
L-BFGS optimisation of an edit matrix against a KL budget. Both are *optimised* scalars or
matrices, not one global alpha.

So this script holds the direction fixed and lets the optimiser choose how hard to remove it
at each of the 48 places, under an explicit interference budget:

    minimise   refusal_margin + lam * KL(harmless) + l2 * sum(w^2)
    over       w in R^48   (one strength per place, init 0 = untouched model)

  refusal_margin  mean over harmful prompts of  logit("I cannot") - logit("Sure")
                  at the first answer position, thinking OFF. Lower is less refusal.
  KL(harmless)    mean over harmless prompts of KL(edited || pristine) on the next-token
                  distribution, averaged over the first N positions. This is the
                  interference term, measured on prompts whose behaviour must not move.
  l2              keeps the edit small; report rel_change so the dose stays comparable with
                  the public builds' 2.5-3.1% / 0.5-17.3% figures.

Stages:
  optim   run the optimiser, write {w, history, kl, margin} to /out/r18/opt.json
  gate    load pristine, apply the optimised w at generation time (pre-norm endpoint, the
          operator a weight edit implements), generate the 12 held-out harmful prompts at
          4096 greedy with thinking OFF, write harness-shaped transcript for the judge
  Both stages are separate so a failed optimiser does not burn the generation pass.

Honest limits, stated before the run:
  - The margin term is a *proxy* for refusal. A model can lower it and still refuse in
    prose. The judge is the only thing that decides, exactly as in every earlier rung.
  - 48 free parameters optimised on 20 harmful prompts will overfit them. The gate runs on
    the 12 held-out prompts, which the optimiser never sees, so the number that counts is
    out-of-sample. If the gate passes on prompts the optimiser never saw, that is real.
  - This is not Heretic's implementation. It is the same idea (per-place strengths against
    an interference budget) built here, so the result is the campaign's own, not a
    reproduction claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MODEL_ID = "LiquidAI/LFM3-1.7B"
PLACES = {"operator": "post_operator_norm", "ffn": "post_ffn_norm"}
REFUSE_TOKENS = ["I cannot", "I can't", "I'm sorry", "Sorry"]
COMPLY_TOKENS = ["Sure", "Here", "To "]


def _enc(tok, prompt, device, thinking=False):
    kw = {} if thinking else {"enable_thinking": False}
    out = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", **kw)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(device)


def _flat_places(model):
    places = []
    for li, layer in enumerate(model.model.layers):
        for place, suffix in PLACES.items():
            mod = getattr(layer, suffix, None)
            if mod is not None:
                places.append((f"L{li}.{place}", mod))
    return places


def _first_token_ids(tok, texts):
    ids = []
    for t in texts:
        row = tok.encode(t, add_special_tokens=False)
        if row:
            ids.append(row[0])
    return ids


def stage_optim(a) -> dict:
    from config import load_config
    from eval_split import build_split
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(a.config)
    dev = a.device
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    h, g = list(DEFAULT_HARMFUL), list(DEFAULT_HARMLESS)
    n_test = int(getattr(cfg, "n_test_prompts", 12) or 12)
    split = build_split(h[:len(g)], g, train_size=len(g) - 2 * n_test, tune_size=n_test,
                        test_size=n_test, seed=cfg.eval_split_seed)
    opt_harmful = list(dict.fromkeys(h[: len(h) - 2 * n_test]))[: a.n_harmful]
    opt_harmless = g[: a.n_harmless]
    held_out = set(split.test)
    opt_harmful = [p for p in opt_harmful if p not in held_out]
    opt_harmless = [p for p in opt_harmless if p not in held_out]
    print(f"optimiser sees {len(opt_harmful)} harmful / {len(opt_harmless)} harmless; "
          f"gate holds out {len(held_out)}", flush=True)

    ref_ids = _first_token_ids(tok, REFUSE_TOKENS)
    comp_ids = _first_token_ids(tok, COMPLY_TOKENS)

    blob = torch.load(a.dirs, map_location="cpu", weights_only=False)
    dirs = blob["dirs"][a.position]

    places = _flat_places(model)
    keys = [k for k, _ in places if k in dirs]
    print(f"places {len(keys)}", flush=True)

    # one learnable strength per place, init 0.0 = pristine
    w = torch.zeros(len(keys), device=dev, dtype=torch.float32, requires_grad=True)
    res = {k: dirs[k].to(device=dev, dtype=torch.bfloat16) for k in keys}

    def apply_hooks():
        handles = []
        for i, (k, mod) in enumerate((kv for kv in places if kv[0] in res)):
            r = res[k]

            def _h(_m, _i, out, r=r, i=i):
                return out - (w[i].to(out.dtype) * (out @ r).unsqueeze(-1) * r)

            handles.append(mod.register_forward_hook(_h))
        return handles

    def batch(prompts, device=dev):
        seqs = [_enc(tok, p, device)[0] for p in prompts]
        L = max(s.shape[0] for s in seqs)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
        am = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
        for j, s in enumerate(seqs):
            ids[j, : s.shape[0]] = s
            am[j, : s.shape[0]] = 1
        return ids, am

    hids, ham = batch(opt_harmful)
    gids, gam = batch(opt_harmless)

    def refusal_margin():
        handles = apply_hooks()
        try:
            out = model(input_ids=hids, attention_mask=ham).logits[:, -1, :].float()
        finally:
            for x in handles:
                x.remove()
        ref = out[:, ref_ids].max(dim=-1).values
        comp = out[:, comp_ids].max(dim=-1).values
        return (ref - comp).mean()

    def kl_harmless():
        # KL(edited || pristine) on the first positions, both computed in the same pass pair
        handles = apply_hooks()
        try:
            logits_e = model(input_ids=gids, attention_mask=gam).logits.float()
        finally:
            for x in handles:
                x.remove()
        with torch.no_grad():
            logits_b = model(input_ids=gids, attention_mask=gam).logits.float()
        n = min(a.kl_positions, logits_e.shape[1] - 1)
        le = logits_e[:, :n, :].log_softmax(-1)
        lb = logits_b[:, :n, :].log_softmax(-1)
        kl = (le.exp() * (le - lb)).sum(-1)[gam[:, :n].bool()]
        return kl.mean()

    opt = torch.optim.Adam([w], lr=a.lr)
    hist = []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        m = refusal_margin()
        kl = kl_harmless()
        loss = m + a.lam * kl + a.l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(0.0, a.max_w)
        if step % max(1, a.steps // 12) == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": round(float(loss), 4),
                   "margin": round(float(m), 4), "kl": round(float(kl), 6),
                   "w_mean": round(float(w.mean()), 4), "w_max": round(float(w.max()), 4)}
            hist.append(rec)
            print(rec, flush=True)

    rec_w = {k: round(float(w[i]), 4) for i, k in enumerate(keys)}
    out = {"w": rec_w, "history": hist, "lam": a.lam, "l2": a.l2,
           "steps": a.steps, "position": a.position, "dirs": a.dirs,
           "n_opt_harmful": len(opt_harmful), "n_opt_harmless": len(opt_harmless),
           "held_out": sorted(held_out)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return out


def stage_gate(a) -> int:
    """Apply the optimised per-place strengths and generate the held-out arm."""
    from config import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hookgate import _held_out, generate  # reuse the same prompt split and loop

    cfg = load_config(a.config)
    dev = a.device
    blob = torch.load(a.opt, map_location="cpu", weights_only=False)
    wmap = blob["w"]
    dblob = torch.load(a.dirs, map_location="cpu", weights_only=False)
    dirs = dblob["dirs"][blob.get("position", a.position)]

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    prompts = _held_out(cfg)
    max_new = a.max_new or int(getattr(cfg, "gate_refusal_max_new_tokens", 1024) or 1024)

    places = _flat_places(model)
    handles = []
    used = {}
    for k, mod in places:
        if k not in dirs or k not in wmap:
            continue
        s = float(wmap[k])
        if s <= 1e-6:
            continue
        used[k] = s
        r = dirs[k].to(device=dev, dtype=torch.bfloat16)

        def _h(_m, _i, out, r=r, s=s):
            return out - (s * (out @ r).unsqueeze(-1) * r)

        handles.append(mod.register_forward_hook(_h))
    print(f"applied {len(used)} places, w range {min(used.values()):.3f}-{max(used.values()):.3f}",
          flush=True)
    try:
        texts = generate(model, tok, prompts, dev, max_new, dirs=None,
                         strength=0.0, where="post", thinking=False)
    finally:
        for h in handles:
            h.remove()

    gens = [{"prompt": p, "response": t, "arm": "hook", "gate": "refusal"}
            for p, t in zip(prompts, texts)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": a.config, "thinking": False,
                               "optimised_w": used, "window": max_new,
                               "generations": gens}, indent=1))
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["optim", "gate"])
    ap.add_argument("--config", default="models/lfm3-1.7b-r8.yaml")
    ap.add_argument("--dirs", default="/out/r9/r9-directions.pt")
    ap.add_argument("--position", default="mean", choices=["mean", "last"])
    ap.add_argument("--out", default="/out/r18/opt.json")
    ap.add_argument("--opt", default="/out/r18/opt.json")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--max-w", type=float, default=2.0)
    ap.add_argument("--kl-positions", type=int, default=16)
    ap.add_argument("--n-harmful", type=int, default=20)
    ap.add_argument("--n-harmless", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.stage == "gate":
        return stage_gate(a)
    stage_optim(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
