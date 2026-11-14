"""Optimise per-place removal strengths against an interference budget, then bake them in.

Why. Rungs 10-17 measured one object: a single harvested direction, projected out at ONE
strength, applied identically to every place. The published builds that clear the bar do not
do that — Heretic discloses per-layer numbers that differ from each other (attn.o_proj
1.19-1.49, peak 14.92, reach 13.28; mlp.down_proj 1.09-1.23, peak 17.78, reach 11.61), and
rSalone's card describes L-BFGS optimisation of an edit matrix against a KL budget. So hold
the direction fixed and let an optimiser choose how hard to remove it at each of the 48 places:

    minimise   refusal_margin + lam * KL(harmless) + l2 * sum(w^2)      over w in R^48

  refusal_margin  logit max("I cannot"/"I can't"/"I'm sorry"/"Sorry"/"I won't")
                  - logit max("Sure"/"Here"/"To "/"1.") at the first answer position,
                  thinking OFF. Lower = the model is less inclined to open with a refusal.
  KL(harmless)    KL(edited || pristine) over the first positions on harmless prompts, i.e.
                  measured on prompts whose behaviour must not move (the interference pool).
  l2              keeps the edit small; rel_change is reported so the dose stays comparable
                  with the public builds' 2.5-3.1% (Heretic) and 0.5-17.3% (rSalone).

Endpoint. Everything here is the PRE-norm endpoint: the hook projects the branch's raw write
along a = normalize(gamma * r), so x' = x - s (x.a) a. That is a linear map on x, so a weight
edit implements it EXACTLY:  W' = W - s a (a^T W).  The post-norm endpoint has no faithful
linear bake (rung 9: F1 0.998 then 0.99997) because RMSNorm's 1/rms(x) depends on x after the
edit. Optimising at an endpoint you cannot ship would repeat that mistake.

Stages, each separately runnable so a failed step does not burn the expensive one:
  optim   fit w, write {w, history} to /out/r18/opt.json
  gate    apply the optimised w at generation time on the 12 held-out harmful prompts
  bake    write a real checkpoint with W' = W - s a (a^T W) into /out/r18/baked
  verify  load the BAKED checkpoint with no hooks at all and generate the same prompts

`gate` says the operator would work if shipped. `verify` says the shipped artifact works, and
`verify` is the stage that decides whether a model was produced.

Honest limits, stated before the run:
  - The margin term is a proxy. A model can lower it and still refuse in prose; the judge
    decides, as in every earlier rung.
  - 48 free parameters fitted on 20 prompts will overfit them. The gate runs on 12 held-out
    prompts the optimiser never sees, so the number that counts is out-of-sample.
  - This is not Heretic's implementation. Same idea, built here, so the result is the
    campaign's own and not a reproduction claim.
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
NORMS = {"operator": "post_operator_norm", "ffn": "post_ffn_norm"}
REFUSE_TOKENS = ["I cannot", "I can't", "I'm sorry", "Sorry", "I won't"]
COMPLY_TOKENS = ["Sure", "Here", "To ", "1."]


# ------------------------------------------------------------------ plumbing
def _enc(tok, prompt, device, thinking=False):
    kw = {} if thinking else {"enable_thinking": False}
    out = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", **kw)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(device)


def _proj_module(layer, place: str):
    """The matrix that writes into the residual stream before this norm."""
    if place == "ffn":
        return layer.feed_forward.w2, "feed_forward.w2.weight"
    if getattr(layer, "is_attention_layer", False):
        return layer.self_attn.o_proj, "self_attn.o_proj.weight"
    return layer.conv.out_proj, "conv.out_proj.weight"


def _axis(layer, place: str, r: torch.Tensor, device) -> torch.Tensor:
    """a = normalize(gamma * r): the pre-norm axis a weight edit removes along."""
    g = getattr(layer, NORMS[place]).weight.detach().to(device=device, dtype=torch.float32)
    a = g * r.float()
    return (a / a.norm().clamp(min=1e-8)).to(dtype=torch.bfloat16)


def _first_ids(tok, texts):
    ids = []
    for t in texts:
        row = tok.encode(t, add_special_tokens=False)
        if row:
            ids.append(row[0])
    return ids


def _batch(tok, prompts, device):
    seqs = [_enc(tok, p, device)[0] for p in prompts]
    L = max(s.shape[0] for s in seqs)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
    am = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for j, s in enumerate(seqs):
        ids[j, : s.shape[0]] = s
        am[j, : s.shape[0]] = 1
    return ids, am


def _last_pos(am):
    """Right padding means logits[:, -1] is a pad slot. Gather the last real token."""
    return (am.sum(1) - 1).clamp(min=0)


def _opt_pools(cfg, n_harmful, n_harmless):
    """Prompts the optimiser may see: everything the held-out gate will not use."""
    from hookgate import _held_out
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    held = set(_held_out(cfg))
    seen = lambda pool: [p for p in pool if p not in held]  # noqa: E731
    return seen(DEFAULT_HARMFUL)[:n_harmful], seen(DEFAULT_HARMLESS)[:n_harmless], held


def _register(model, dirs, wmap, device, tally=None):
    """Pre-norm hooks whose strength is read from wmap (dict of floats or live tensors)."""
    handles, keys = [], []
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        for place in NORMS:
            key = f"L{li}.{place}"
            if key not in dirs or key not in wmap:
                continue
            r = dirs[key].to(device=device, dtype=torch.float32)
            a = _axis(layer, place, r, device)
            s = wmap[key]
            mod = getattr(layer, NORMS[place])
            keys.append(key)

            def _p(_m, args, a=a, s=s):
                x = args[0]
                if tally is not None:
                    tally["edited"] += 1
                    if x.shape[1] == 1:
                        tally["decode_passes"] += 1
                    else:
                        tally["prompt_passes"] += 1
                term = (x.float() @ a.float()).to(x.dtype)
                ss = s.to(term.dtype) if torch.is_tensor(s) else s
                return (x - (ss * term).unsqueeze(-1) * a,) + tuple(args[1:])

            handles.append(mod.register_forward_pre_hook(_p))
    return handles, keys


@torch.no_grad()
def _gen(model, tok, prompts, dev, max_new, tally=None):
    from hookgate import _encode
    out = []
    for p in prompts:
        ids = _encode(tok, p, dev, thinking=False)
        gen = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out.append(tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True))
    if tally is not None:
        print(f"  tally: prompt_passes {tally['prompt_passes']} "
              f"decode_passes {tally['decode_passes']} edited {tally['edited']}", flush=True)
    return out


def _write(path, config, arm, texts, prompts, meta):
    gens = [{"prompt": p, "response": t, "arm": arm, "gate": "refusal"}
            for p, t in zip(prompts, texts)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"config": config, "thinking": False,
                                      "generations": gens, **meta}, indent=1))
    lens = [len(t) for t in texts]
    print(f"  {arm}: {len(texts)} generations, mean chars {sum(lens)//max(1,len(lens))}",
          flush=True)
    print(f"wrote {path}")


# ------------------------------------------------------------------ stages
def stage_optim(a) -> int:
    from config import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(a.config)
    dev = a.device
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    opt_harmful, opt_harmless, held = _opt_pools(cfg, a.n_harmful, a.n_harmless)
    print(f"optimiser sees {len(opt_harmful)} harmful / {len(opt_harmless)} harmless; "
          f"gate holds out {len(held)}", flush=True)

    blob = torch.load(a.dirs, map_location="cpu", weights_only=False)
    dirs = blob["dirs"][a.position]
    keys = [f"L{li}.{p}" for li in range(N_LAYERS) for p in NORMS if f"L{li}.{p}" in dirs]
    print(f"places {len(keys)}", flush=True)

    w = torch.zeros(len(keys), device=dev, dtype=torch.float32, requires_grad=True)
    handles, live = _register(model, dirs, {k: w[i] for i, k in enumerate(keys)}, dev)
    assert live == keys, (len(live), len(keys))

    ref_ids = _first_ids(tok, REFUSE_TOKENS)
    comp_ids = _first_ids(tok, COMPLY_TOKENS)
    hids, ham = _batch(tok, opt_harmful, dev)
    gids, gam = _batch(tok, opt_harmless, dev)
    hlast = _last_pos(ham)
    rows = torch.arange(len(opt_harmful), device=dev)

    with torch.no_grad():
        base_lp = model(input_ids=gids, attention_mask=gam).logits.float()[:, :-1].log_softmax(-1)

    def refusal_margin():
        out = model(input_ids=hids, attention_mask=ham).logits.float()[rows, hlast]
        ref = out[:, ref_ids].max(dim=-1).values
        comp = out[:, comp_ids].max(dim=-1).values
        return (ref - comp).mean()

    def kl_harmless():
        lp = model(input_ids=gids, attention_mask=gam).logits.float()[:, :-1].log_softmax(-1)
        n = min(a.kl_positions, lp.shape[1])
        kl = (lp[:, :n].exp() * (lp[:, :n] - base_lp[:, :n])).sum(-1)
        return kl[gam[:, :n].bool()].mean()

    opt = torch.optim.Adam([w], lr=a.lr)
    hist = []
    for step in range(a.steps + 1):
        opt.zero_grad(set_to_none=True)
        m = refusal_margin()
        kl = kl_harmless()
        loss = m + a.lam * kl + a.l2 * (w ** 2).sum()
        if step < a.steps:
            loss.backward()
            opt.step()
            with torch.no_grad():
                w.clamp_(0.0, a.max_w)
        if step % max(1, a.steps // 10) == 0 or step == a.steps:
            rec = {"step": step, "loss": round(float(loss), 4),
                   "margin": round(float(m), 4), "kl": round(float(kl), 6),
                   "w_mean": round(float(w.mean()), 4), "w_max": round(float(w.max()), 4)}
            hist.append(rec)
            print(rec, flush=True)
    for h in handles:
        h.remove()

    out = {"w": {k: round(float(w[i]), 4) for i, k in enumerate(keys)},
           "history": hist, "lam": a.lam, "l2": a.l2, "steps": a.steps,
           "position": a.position, "dirs": a.dirs, "endpoint": "pre",
           "n_opt_harmful": len(opt_harmful), "n_opt_harmless": len(opt_harmless),
           "held_out": sorted(held)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    wv = sorted(out["w"].values())
    print(f"w quantiles: min {wv[0]:.3f} med {wv[len(wv)//2]:.3f} max {wv[-1]:.3f}")
    print(f"wrote {a.out}")
    return 0


def stage_gate(a) -> int:
    from config import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hookgate import _held_out

    cfg = load_config(a.config)
    dev = a.device
    opt = json.loads(Path(a.opt).read_text())
    wmap = opt["w"]
    dirs = torch.load(a.dirs, map_location="cpu", weights_only=False)["dirs"][opt["position"]]

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    tally = {"prompt_passes": 0, "decode_passes": 0, "edited": 0}
    handles, used = _register(model, dirs, wmap, dev, tally)
    print(f"applied {len(used)} places at the pre-norm endpoint, thinking OFF", flush=True)
    prompts = _held_out(cfg)
    try:
        texts = _gen(model, tok, prompts, dev, a.max_new, tally)
    finally:
        for h in handles:
            h.remove()
    _write(a.out, a.config, "hook", texts, prompts, {"tally": tally, "w": wmap,
                                                     "endpoint": "pre"})
    return 0


def stage_bake(a) -> int:
    """Write W' = W - s a (a^T W) into a servable checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import load_config

    opt = json.loads(Path(a.opt).read_text())
    wmap = opt["w"]
    dirs = torch.load(a.dirs, map_location="cpu", weights_only=False)["dirs"][opt["position"]]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32,
                                                 trust_remote_code=True)
    rows = []
    with torch.no_grad():
        for li in range(N_LAYERS):
            layer = model.model.layers[li]
            for place in NORMS:
                key = f"L{li}.{place}"
                if key not in dirs or key not in wmap or float(wmap[key]) <= 1e-6:
                    continue
                s = float(wmap[key])
                a_vec = _axis(layer, place, dirs[key], torch.device("cpu")).float()
                proj, wname = _proj_module(layer, place)
                W = proj.weight.data
                before = W.float().norm().item()
                a_col = a_vec.unsqueeze(1)                       # (d,1)
                upd = (a_col @ (a_col.T @ W.float()))            # rank-1, exact
                W.sub_((s * upd).to(W.dtype))
                after = W.float().norm().item()
                rows.append({"key": key, "tensor": wname, "s": round(s, 4),
                             "rel_change": round(abs(after - before) / max(before, 1e-9), 6)})
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    cfg_src = Path(a.config)
    (out / "bake-note.txt").write_text(
        "rank-1 pre-norm ablation, W' = W - s a (a^T W), a = normalize(gamma * r)\n"
        f"config {cfg_src.name}, optimiser {a.opt}, places {len(rows)}\n")
    rel = [r["rel_change"] for r in rows]
    report = {"n_places": len(rows), "endpoint": "pre", "opt": a.opt,
              "rel_change": {"mean": round(sum(rel) / max(1, len(rel)), 6),
                             "max": round(max(rel) if rel else 0.0, 6)},
              "places": rows, "w": wmap}
    (out / "bake-report.json").write_text(json.dumps(report, indent=1))
    print(f"baked {len(rows)} places into {out}; rel_change mean "
          f"{report['rel_change']['mean']} max {report['rel_change']['max']}")
    return 0


def stage_verify(a) -> int:
    """Load the baked checkpoint with NO hooks. This is the shipped-artifact test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import load_config
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hookgate import _held_out

    cfg = load_config(a.config)
    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.out_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.out_dir, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    prompts = _held_out(cfg)
    texts = _gen(model, tok, prompts, dev, a.max_new)
    _write(a.out, a.config, "baked", texts, prompts, {"baked_from": a.out_dir})
    rep = json.loads((Path(a.out_dir) / "bake-report.json").read_text())
    print(f"no hooks registered; rel_change mean {rep['rel_change']['mean']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["optim", "gate", "bake", "verify"])
    ap.add_argument("--config", default="models/lfm3-1.7b-r8.yaml")
    ap.add_argument("--dirs", default="/out/r9/r9-directions.pt")
    ap.add_argument("--position", default="mean", choices=["mean", "last"])
    ap.add_argument("--opt", default="/out/r18/opt.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default="/out/r18/baked")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--max-w", type=float, default=2.0)
    ap.add_argument("--kl-positions", type=int, default=16)
    ap.add_argument("--n-harmful", type=int, default=20)
    ap.add_argument("--n-harmless", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out = a.out or f"/out/r18/{a.stage}.json"
    if a.stage == "optim":
        return stage_optim(a)
    if a.stage == "gate":
        return stage_gate(a)
    if a.stage == "bake":
        return stage_bake(a)
    return stage_verify(a)


if __name__ == "__main__":
    sys.exit(main())
