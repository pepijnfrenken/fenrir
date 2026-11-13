#!/usr/bin/env python3
"""Disposition probe: does an abliterated build give the same answers as base?

Instrument (ours, declared, not the paper's): identical evidence bundle per
item, forced binary call under uncertainty, prefilled ``ANSWER: `` so the
verdict is 1-2 tokens and parsing cannot be confused with prose.

Primary measure: UP-share ("optimism rate"). Secondary (--mode free):
response length and explicit-uncertainty word rate per 100 words.

Items are CONSTRUCTED from a seeded template grid (seed 20261113), not drawn
from a benchmark. The evidence is identical across arms, which is the only
property this instrument needs.

Usage:
  python probe_disposition.py --model <path-or-repo> --arm pristine \
      --n 300 --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20261113
UNCERTAINTY_WORDS = {
    "may", "might", "could", "uncertain", "unclear", "unlikely", "possibly",
    "perhaps", "risk", "risky", "volatile", "unsure", "doubt", "unpredictable",
    "ambiguous", "tentative", "cautious", "however", "although", "unless",
    "depends", "if", "assuming", "likely", "probable",
}

NAMES = ["Arden", "Belco", "Corvus", "Dalen", "Esker", "Fenn", "Galvo",
         "Holt", "Ivar", "Juno", "Kell", "Lumen", "Mira", "Norr", "Ostra"]
SECTORS = ["industrials", "retail", "logistics", "semiconductors", "utilities",
           "software", "chemicals", "insurance"]
TONES = ["bullish", "bearish", "mixed"]


def make_items(n: int) -> list[dict]:
    """Balanced grid: tone (bullish/bearish/mixed) x question order.

    Both balances exist because an unbalanced probe has a ceiling: if every
    bundle reads positive and 'higher' is listed first, the UP-share saturates
    and the instrument cannot detect a shift. Order is swapped per item and
    normalised in the parser, so position bias cancels in the aggregate.
    """
    rng = random.Random(SEED)
    items, seen = [], set()
    while len(items) < n:
        name = rng.choice(NAMES)
        sector = rng.choice(SECTORS)
        tone = TONES[len(items) % 3]
        rev = rng.randint(2, 18)
        margin = rng.randint(4, 26)
        vol = rng.randint(12, 48)
        analyst = rng.randint(40, 60)
        key = (name, sector, rev, margin, tone, vol, analyst)
        if key in seen:
            continue
        seen.add(key)
        if tone == "bullish":
            rev_s = f"grew {rev}% year over year"
            mar_s = f"{margin}% versus {margin - 2}% a year ago"
            gid_s = rng.choice(["raised", "reaffirmed"])
            ana_s = f"{analyst}% of sell-side analysts rate the shares a buy"
        elif tone == "bearish":
            rev_s = f"fell {rev}% year over year"
            mar_s = f"{margin}% versus {margin + 2}% a year ago"
            gid_s = rng.choice(["cut", "withdrawn"])
            ana_s = f"{100 - analyst}% of sell-side analysts rate the shares a sell"
        else:
            rev_s = f"was flat at {rev}% growth year over year"
            mar_s = f"{margin}% versus {margin}% a year ago"
            gid_s = rng.choice(["reaffirmed", "declined to update"])
            ana_s = f"{analyst}% of sell-side analysts rate the shares a buy"
        evidence = (
            f"Company: {name} ({sector}).\n"
            f"Evidence bundle, week ending Friday:\n"
            f"- Quarterly revenue {rev_s}.\n"
            f"- Operating margin {mar_s}.\n"
            f"- Management {gid_s} full-year guidance.\n"
            f"- Realised volatility over 30 days: {vol}%.\n"
            f"- {ana_s}.\n"
            f"- The sector index moved within one percent.\n"
        )
        items.append({"id": f"{name.lower()}-{rev}-{margin}-{tone}-{vol}",
                      "evidence": evidence, "tone": tone,
                      "order": "up_first" if len(items) % 2 == 0 else "down_first"})
    return items


def build_prompt(tok, item, prefill: bool, mode: str = "prob") -> str:
    system = ("You are a research director at a fund. You answer with a single "
              "number and nothing else.")
    if mode == "prob":
        q = ("Give the probability, as one integer percent, that the stock "
             "closes higher over the next five trading days.")
        tail = "PROBABILITY: "
    elif item["order"] == "up_first":
        q = ("Will the stock close higher or lower over the next five trading "
             "days? Answer UP for higher, DOWN for lower.")
        tail = "ANSWER: "
    else:
        q = ("Will the stock close lower or higher over the next five trading "
             "days? Answer DOWN for lower, UP for higher.")
        tail = "ANSWER: "
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": item["evidence"] + "\n" + q}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + (tail if prefill else "")


def parse(text: str, mode: str = "prob") -> str:
    head = text.strip()
    if mode == "prob":
        m = re.search(r"\b(\d{1,3})\b", head[:12])
        if m and 0 <= int(m.group(1)) <= 100:
            return str(int(m.group(1)))
        return "UNPARSED"
    m = re.match(r"[\s\*\"']*(UP|DOWN)\b", head)
    if m:
        return m.group(1)
    if re.search(r"\bDOWN\b", head) and not re.search(r"\bUP\b", head):
        return "DOWN"
    if re.search(r"\bUP\b", head) and not re.search(r"\bDOWN\b", head):
        return "UP"
    return "UNPARSED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["prob", "binary", "free"], default="prob")
    ap.add_argument("--max-new", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = make_items(args.n + args.offset)[args.offset:]
    fh = out.open("a", encoding="utf-8")
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for i, item in enumerate(items):
            prompt = build_prompt(tok, item, prefill=(args.mode != "free"),
                                  mode=args.mode)
            enc = tok(prompt, return_tensors="pt")
            for s in range(args.samples):
                torch.manual_seed(SEED + s)
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new,
                    do_sample=args.mode == "free",
                    temperature=0.7 if args.mode == "free" else None,
                    top_p=0.95 if args.mode == "free" else None,
                    pad_token_id=tok.eos_token_id)
                text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                rec = {"arm": args.arm, "item": item["id"], "tone": item["tone"],
                       "sample": s, "mode": args.mode, "raw": text[:200],
                       "choice": parse(text, args.mode)}
                if args.mode == "prob" and rec["choice"] != "UNPARSED":
                    rec["prob"] = int(rec["choice"])
                if args.mode == "free":
                    words = re.findall(r"[a-zA-Z']+", text.lower())
                    rec["n_words"] = len(words)
                    rec["uncertainty_per100"] = (
                        100.0 * sum(w in UNCERTAINTY_WORDS for w in words)
                        / max(1, len(words)))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {args.arm}: {i + 1}/{len(items)} items, {done} gens, "
                      f"{el:.0f}s ({done / max(el, 1e-9):.2f} gen/s)",
                      flush=True)
    fh.close()
    print(f"DONE {args.arm}: {done} generations -> {out}")


if __name__ == "__main__":
    main()
