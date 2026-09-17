#!/usr/bin/env python3
"""jev-lane / lens round 3 — steering bridge: inject dref at block 11 and read
the lens.

For each (item, alpha): add alpha * dref_raw_lfm25[11] to the residual stream at
the LAST PROMPT POSITION right after block 11 (== the state the lens reads at
hs12). The hook is shape-guarded so it fires on the prefill only — the captured
readout AND the generation both run from the same steered decision state.

Conditions: benign items  -> alpha in {0,1,2,4,8,16,32} + one random-direction
control at 16 (same magnitude, random unit vector). Harmful items -> alpha in
{0,-1,-2,-4,-8,-16,-32} (pushing away from refusal, the live-abliteration side).

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
INJ_LAYER = 11  # block index; its output is hs12 — the legible-window peak
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
    args = ap.parse_args()

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
    v_raw = torch.tensor(d["dref_raw_lfm25"][INJ_LAYER])  # [2048] float32 mean-diff
    print(f"[steer] dref raw norm @L{INJ_LAYER} = {float(v_raw.norm()):.4f}", flush=True)
    rng = np.random.default_rng(0)
    randvec = rng.standard_normal(v_raw.shape[0]).astype(np.float32)
    randvec = torch.tensor(randvec / np.linalg.norm(randvec))

    conds = []
    for it in items:
        if it["condition"] == "answer":
            for a in (0, 1, 2, 4, 8, 16, 32):
                conds.append((it, f"a{a:+d}", float(a), False))
            conds.append((it, "rand+16", 16.0, True))
        else:  # refuse
            for a in (0, -1, -2, -4, -8, -16, -32):
                conds.append((it, f"a{a:+d}", float(a), False))
    print(f"[steer] {len(conds)} captures planned", flush=True)

    recs = []
    for n, (it, cname, alpha, is_rand) in enumerate(conds, 1):
        msgs = [{"role": "user", "content": it["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        pos = len(ids) - 1
        inp = torch.tensor([ids]).to(device)
        vec = ((randvec if is_rand else v_raw).to(device) * alpha)

        state = {"h_norm": None}

        def hook(mod, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if o.shape[1] <= pos:  # decode step — prefill-only steering
                return out
            o2 = o.clone()
            o2[0, pos, :] = o2[0, pos, :] + vec.to(o2.dtype)
            state["h_norm"] = float(o2[0, pos].float().norm())
            return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2

        hk = base.layers[INJ_LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model(inp, output_hidden_states=True, use_cache=False)
                layers_out = {str(k): lens_toks(out.hidden_states[k][0, pos].float())
                              for k in range(1, L + 1)}
                gen = model.generate(inp, max_new_tokens=args.gen, do_sample=False)
        finally:
            hk.remove()
        rec = {
            "id": it["id"], "condition": it["condition"], "task": it.get("task"),
            "prompt": it["prompt"], "cond": cname, "alpha": alpha, "rand": is_rand,
            "inj_layer": INJ_LAYER, "delta_norm": float((vec.float()).norm()),
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

    # sanity: one benign + one harmful item, hs12 at the extreme alphas
    for cid in ("A0", "R0"):
        sel = [r for r in recs if r["id"] == cid]
        print(f"\n=== {cid} ({sel[0]['condition']}): {sel[0]['prompt'][:80]!r}")
        for r in sel:
            print(f"  alpha={r['alpha']:>5} | hs12: {' | '.join(r['layers']['12'][:8])}")
            print(f"        gen: {r['gen_prefix'][:90]!r}")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
