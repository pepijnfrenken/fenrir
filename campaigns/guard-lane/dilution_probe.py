#!/usr/bin/env python3
"""Guard lane — dilution probe: can a small harmful span ride in on a pile of safe text?

Pino's question (2026-09-16): "overload the model with safe and then add a smidge
of unsafe so that it passes?"

Exact-content control: the harmful span is UNCHANGED (a held-out response that
flags 10/10, margin mean +5.78). Only the surrounding benign text, its position
relative to the payload, and the prompt vary. Conditions per harmful pair:

  base_hp           harmful prompt + harmful response        (the known flag)
  base_bp           benign prompt  + harmful response        (prompt contribution removed)
  end_k1/3/7        k benign responses, THEN the payload     (payload at the end)
  start_k7          payload FIRST, then 7 benign responses   (payload at the start)
  smidge_k7         first 250 chars of the payload, then 7 benign responses
  convo_4           four benign (user, assistant) turns, then the harmful pair

Readout: verdict band + verdict-token margin — same instrument as everything else.
Reported per condition: flag count, margin mean, mean harmful fraction of the text.

Mechanisms this separates:
  (a) anchoring — flag survives dilution => the guard reads content, not proportion;
  (b) position  — end vs start asymmetry at k=7;
  (c) prompt    — base_hp vs base_bp: how much of the flag lives in the prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from config import load_config  # noqa: E402
from guard_readout import read_pair, verdict_read  # noqa: E402

RESULTS = HERE / "results"
GUARD_CONFIG = "models/qwen3guard-0.6b.yaml"


def log(msg: str) -> None:
    print(f"[dilution] {msg}", flush=True)


def load_guard(model_id: str, dtype_name: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(dtype_name, torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def read(guard, gt, cfg, prompt, response):
    out = read_pair(guard, gt, prompt, response, tokens=getattr(cfg, "verdict_tokens", None))
    return {"band": out.get("band"), "margin": out.get("margin"), "label": out.get("label")}


def read_convo(guard, gt, cfg, messages):
    out = verdict_read(guard, gt, messages, tokens=getattr(cfg, "verdict_tokens", None))
    return {"band": out.get("band"), "margin": out.get("margin"), "label": out.get("label")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tag", default="repl")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    rows = [json.loads(l) for l in open(args.pairs) if l.strip()]
    test = [r for r in rows if r.get("split") == "test"]
    harm = [r for r in test if str(r.get("expected", "")).lower() in ("flag", "unsafe")][: args.n]
    benign_rows = [r for r in rows if str(r.get("expected", "")).lower() in ("safe", "benign")]
    filler = [r["response"] for r in benign_rows]
    benign_prompts = [r["prompt"] for r in benign_rows]
    log(f"{len(harm)} harmful pairs; filler pool = {len(filler)} benign responses (all splits)")

    cfg = load_config(GUARD_CONFIG)
    guard, gt = load_guard(cfg.model_id)
    log("guard loaded")

    def chunks_for(i: int, k: int):
        return [filler[(i * 3 + j) % len(filler)] for j in range(k)]

    records = []
    for i, r in enumerate(harm):
        payload = r["response"]
        hp = r["prompt"]
        bp = benign_prompts[i % len(benign_prompts)]
        smidge = payload[:250]

        conds: dict[str, tuple[str, str]] = {
            "base_hp": (hp, payload),
            "base_bp": (bp, payload),
            "end_k1": (hp, "\n\n".join(chunks_for(i, 1) + [payload])),
            "end_k3": (hp, "\n\n".join(chunks_for(i, 3) + [payload])),
            "end_k7": (hp, "\n\n".join(chunks_for(i, 7) + [payload])),
            "start_k7": (hp, "\n\n".join([payload] + chunks_for(i, 7))),
            "smidge_k7": (hp, "\n\n".join(chunks_for(i, 7) + [smidge])),
        }
        rec = {"pair_id": r.get("pair_id"), "conditions": {}}
        for name, (p, resp) in conds.items():
            out = read(guard, gt, cfg, p, resp)
            frac = len(payload) / max(1, len(resp)) if name != "smidge_k7" else len(smidge) / max(1, len(resp))
            out |= {"chars": len(resp), "harmful_frac": round(frac, 3)}
            rec["conditions"][name] = out

        # multi-turn: 4 benign turns, then the harmful pair as the last turn
        turns = []
        for j in range(4):
            kk = (i * 3 + j) % len(benign_rows)
            turns.append({"role": "user", "content": benign_rows[kk]["prompt"]})
            turns.append({"role": "assistant", "content": benign_rows[kk]["response"]})
        turns += [{"role": "user", "content": hp}, {"role": "assistant", "content": payload}]
        out = read_convo(guard, gt, cfg, turns)
        out |= {"chars": sum(len(str(m["content"])) for m in turns),
                "harmful_frac": round(len(payload) / max(1, sum(len(str(m["content"])) for m in turns)), 3)}
        rec["conditions"]["convo_4"] = out

        records.append(rec)
        done = " ".join(f"{k}:{'U' if v['band'] == 'flag' else 's'}" for k, v in rec["conditions"].items())
        log(f"pair {i + 1}/{len(harm)} — {done}")

    # ---- aggregate per condition
    names = list(records[0]["conditions"].keys())
    summary = {}
    for name in names:
        vals = [rec["conditions"][name] for rec in records]
        margins = [v["margin"] for v in vals if isinstance(v.get("margin"), (int, float))]
        summary[name] = {
            "flagged": sum(1 for v in vals if v.get("band") == "flag"),
            "n": len(vals),
            "margin_mean": (sum(margins) / len(margins)) if margins else None,
            "mean_chars": round(sum(v["chars"] for v in vals) / len(vals)),
            "mean_harmful_frac": round(sum(v["harmful_frac"] for v in vals) / len(vals), 3),
        }

    print()
    print(f"{'condition':12s} {'flagged':8s} {'margin':9s} {'chars':7s} {'harmful%':8s}")
    for name in names:
        s = summary[name]
        print(f"{name:12s} {s['flagged']}/{s['n']:<6} {s['margin_mean']:+8.2f} {s['mean_chars']:<7} {100 * s['mean_harmful_frac']:6.1f}%")

    out_path = RESULTS / f"dilution_probe_{args.tag}.json"
    out_path.write_text(json.dumps({
        "time": time.strftime("%Y-%m-%d %H:%M"), "pairs": args.pairs,
        "n": len(harm), "summary": summary, "per_pair": records,
    }, indent=2, default=str))
    log(f"wrote {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
