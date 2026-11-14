#!/usr/bin/env python3
"""Honest interference: first-token KL and prompt PPL on the HARMLESS pool.

Why this exists (lfm3-1.7b, rung 7b): `collect` measured perplexity_increase and
first_token_kl on the held-out HARMFUL prompts, so a build that stops refusing
fails by construction. The public Heretic build that refused 0/12 and gained
MMLU read KL 11.73 nats on that set; edits that still refused read ~0.3 and
looked cleaner. Interference belongs where behaviour must not move.

This script measures both models on the same harmless prompts and reports:
  kl_first_token   KL(abliterated || pristine) at the last prompt token
  ppl_increase     PPL(abl) / PPL(pristine) - 1 over the prompt text

Usage:
  python interference_check.py --base LiquidAI/LFM3-1.7B \
      --other campaigns/liquidai-lfm3-1.7b/public/heretic-max \
      --other campaigns/liquidai-lfm3-1.7b/mpoa-a2.0-all24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(path_or_id: str, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    local = Path(path_or_id).exists()
    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if local:
        kw["trust_remote_code"] = True
    else:
        kw["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(path_or_id, **kw)
    tok = AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    return model, tok, dev


def _measure(model, tok, dev, prompts, flavor: str):
    """Return (first-token log-softmax per prompt, prompt PPL per prompt)."""
    import torch
    from prompt_format import format_prompt
    first, ppl = [], []
    for p in prompts:
        formatted = format_prompt(tok, p, flavor)
        inp = tok(formatted, return_tensors="pt", truncation=True, max_length=1024).to(dev)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits.float()
        first.append(torch.log_softmax(lg[0, -1], dim=-1).cpu())
        cont = lg[0, 0: lg.shape[1] - 1]
        tokens = inp["input_ids"][0, 1:]
        if cont.shape[0] != tokens.shape[0]:
            ppl.append(float("nan"))
            continue
        chosen = torch.log_softmax(cont, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ppl.append(math.exp(-chosen.sum().item() / max(1, chosen.numel())))
    return first, ppl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--other", action="append", required=True)
    ap.add_argument("--n", type=int, default=10, help="harmless prompts (max 10, as the gate does)")
    ap.add_argument("--flavor", default="chat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from prompts import DEFAULT_HARMLESS

    prompts = list(DEFAULT_HARMLESS)[-args.n:]
    print(f"interference on {len(prompts)} harmless prompts, flavor={args.flavor}")

    bmodel, btok, dev = _load(args.base, torch.bfloat16)
    b_first, b_ppl = _measure(bmodel, btok, dev, prompts, args.flavor)
    del bmodel
    torch.cuda.empty_cache()

    rows = []
    for other in args.other:
        model, tok, dev = _load(other, torch.bfloat16)
        o_first, o_ppl = _measure(model, tok, dev, prompts, args.flavor)
        kls, incs = [], []
        for bf, of, bp, op in zip(b_first, o_first, b_ppl, o_ppl):
            bf, of = bf.to(dev), of.to(dev)
            kls.append(float(torch.nn.functional.kl_div(of, bf, reduction="sum", log_target=True)))
            if bp > 0 and not math.isnan(op):
                incs.append(op / bp - 1.0)
        row = {
            "target": other,
            "kl_first_token_mean": round(sum(kls) / len(kls), 4),
            "kl_max": round(max(kls), 4),
            "ppl_increase_mean": round(sum(incs) / len(incs), 4) if incs else None,
            "n": len(kls),
            "prompt_set": "harmless interference set",
        }
        rows.append(row)
        print(f"{Path(other).name if Path(other).exists() else other}: "
              f"KL {row['kl_first_token_mean']:.4f} (max {row['kl_max']:.4f}), "
              f"PPL {row['ppl_increase_mean']:+.2%}" if incs else "")
        del model
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
