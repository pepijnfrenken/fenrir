#!/usr/bin/env python3
"""jev-lane / lens — steering bridge, parameterized (round 3 = block 11, round 4
= block 9).

For each (item, alpha): add alpha * dref_raw_lfm25[src] to the residual stream at
the LAST PROMPT POSITION right after block `inj_layer`. The hook is shape-guarded
so it fires on the prefill only — the capture AND the generation both run from
the same steered decision state.

Note the capture quirk (documented in the log): recorded `hidden_states` at the
hooked layer is the PRE-injection copy; the effect shows from hs inj+2 onward
(hs inj+1 is the recorded copy for injections at block inj — empirical summary:
read downstream).

Conditions:
  benign  -> alpha in --pos                        (+ one random-dir control @16)
  harmful -> alpha in --neg
  --mismatch adds benign arms injecting dref_raw[--mismatch-src] at --inj-layer.

Outputs are LOCAL (out/ is gitignored) — preview ToS §1(v).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MODEL_DEFAULT = "LiquidAI/LFM2.5-1.2B-Instruct"
DREF_DEFAULT = HERE / "dref_vectors.npz"
ITEMS_DEFAULT = HERE / "items_v0.json"
GEN_TOKENS = 24
TOPK = 12


def clean_tok(tok, tid: int) -> str:
    s = tok.decode([tid])
    s = s.replace("\n", "\\n")
    if s.strip() == "":
        s = "∅" if s == "" else repr(s)[1:-1]
    return s[:24]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--items", default=str(ITEMS_DEFAULT))
    ap.add_argument("--dref", default=str(DREF_DEFAULT))
    ap.add_argument("--tag", default="lfm25")
    ap.add_argument("--gen", type=int, default=GEN_TOKENS)
    ap.add_argument("--inj-layer", type=int, default=11,
                    help="block index whose output gets the injection")
    ap.add_argument("--src-layer", type=int, default=None,
                    help="dref_raw layer to inject (default: inj-layer)")
    ap.add_argument("--pos", default="0,1,2,4,8,16,32", help="benign alphas")
    ap.add_argument("--neg", default="0,-1,-2,-4,-8,-16,-32", help="harmful alphas")
    ap.add_argument("--mismatch", action="store_true",
                    help="add benign arms injecting dref_raw[mismatch_src] at inj-layer")
    ap.add_argument("--mismatch-src", type=int, default=11)
    ap.add_argument("--mismatch-alphas", default="2,8")
    args = ap.parse_args()
    src_layer = args.src_layer if args.src_layer is not None else args.inj_layer

    OUT.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads(Path(args.items).read_text())
             if it["condition"] in ("answer", "refuse")]
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 6)))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = None
    for dt, name in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
            print(f"[steer] loaded as {name} in {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as e:  # pragma: no cover
            print(f"[steer] load as {name} failed: {type(e).__name__}: {e}", flush=True)
    if model is None:
        raise SystemExit("could not load model")
    model.to(device).eval()
    base = model.model
    L = len(base.layers)

    norm_mod = getattr(base, "embedding_norm", None)
    if norm_mod is None:
        raise SystemExit("embedding_norm not found — LFM2 layout changed?")
    W = model.lm_head.weight.float()

    def lens_toks(h: torch.Tensor):
        hn = norm_mod(h.to(norm_mod.weight.dtype)).float()
        _v, idx = torch.topk(hn @ W.T, TOPK)
        return [clean_tok(tok, int(t)) for t in idx]

    d = np.load(args.dref, allow_pickle=True)

    def dref_of(layer: int) -> torch.Tensor:
        return torch.tensor(d["dref_raw_lfm25"][layer])

    vecs = {src_layer: dref_of(src_layer)}
    print(f"[steer] inject at block {args.inj_layer} (hs{args.inj_layer+1} downstream); "
          f"dref src L{src_layer} norm {float(vecs[src_layer].norm()):.4f}", flush=True)
    if args.mismatch:
        vecs[args.mismatch_src] = dref_of(args.mismatch_src)
        print(f"[steer] mismatch arm: dref src L{args.mismatch_src} norm "
              f"{float(vecs[args.mismatch_src].norm()):.4f}", flush=True)
    rng = np.random.default_rng(0)
    rv = rng.standard_normal(vecs[src_layer].shape[0]).astype(np.float32)
    randvec = torch.tensor(rv / np.linalg.norm(rv))

    pos = [float(x) for x in args.pos.split(",") if x.strip()]
    neg = [float(x) for x in args.neg.split(",") if x.strip()]
    mis = [float(x) for x in args.mismatch_alphas.split(",") if x.strip()]
    conds = []
    for it in items:
        if it["condition"] == "answer":
            for a in pos:
                conds.append((it, f"a{a:+.0f}", a, False, src_layer))
            conds.append((it, "rand+16", 16.0, True, src_layer))
            if args.mismatch:
                for a in mis:
                    conds.append((it, f"m{a:+.0f}", a, False, args.mismatch_src))
        else:
            for a in neg:
                conds.append((it, f"a{a:+.0f}", a, False, src_layer))
    print(f"[steer] {len(conds)} captures planned", flush=True)

    recs = []
    for n, (it, cname, alpha, is_rand, src) in enumerate(conds, 1):
        msgs = [{"role": "user", "content": it["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        posn = len(ids) - 1
        inp = torch.tensor([ids]).to(device)
        base_vec = randvec if is_rand else vecs[src]
        vec = base_vec.to(device) * alpha

        state = {"h_norm": None}

        def hook(mod, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if o.shape[1] <= posn:  # decode step — prefill-only steering
                return out
            o2 = o.clone()
            o2[0, posn, :] = o2[0, posn, :] + vec.to(o2.dtype)
            state["h_norm"] = float(o2[0, posn].float().norm())
            return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2

        hk = base.layers[args.inj_layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model(inp, output_hidden_states=True, use_cache=False)
                layers_out = {str(k): lens_toks(out.hidden_states[k][0, posn].float())
                              for k in range(1, L + 1)}
                gen = model.generate(inp, max_new_tokens=args.gen, do_sample=False)
        finally:
            hk.remove()
        rec = {
            "id": it["id"], "condition": it["condition"], "task": it.get("task"),
            "prompt": it["prompt"], "cond": cname, "alpha": alpha, "rand": is_rand,
            "inj_layer": args.inj_layer, "src_layer": src,
            "delta_norm": float((vec.float()).norm()),
            "h_norm_steered": state["h_norm"], "n_blocks": L,
            "layers": layers_out,
            "gen_prefix": tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True),
        }
        recs.append(rec)
        print(f"  [{n}/{len(conds)}] {it['id']:>3} {cname:<7} gen={rec['gen_prefix'][:44]!r}",
              flush=True)

    out_path = OUT / f"steer_{args.tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[steer] wrote {out_path} in {time.time()-t0:.0f}s", flush=True)

    for cid in ("A0", "R0"):
        sel = [r for r in recs if r["id"] == cid]
        print(f"\n=== {cid} ({sel[0]['condition']}): {sel[0]['prompt'][:80]!r}")
        for r in sel:
            print(f"  {r['cond']:>7} | hs12: {' | '.join(r['layers']['12'][:8])}")
            print(f"          {'':<1} hs13: {' | '.join(r['layers']['13'][:8])}")
            print(f"          {'':<1} gen: {r['gen_prefix'][:90]!r}")


if __name__ == "__main__":
    main()
