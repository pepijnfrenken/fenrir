#!/usr/bin/env python3
"""Judge a `collect --transcript` file with the same blind REFUSE/ANSWER/GARBLED
protocol used on the steer tests, grouped by gate and prompt class.

Why: the refusal gate and the coherence gate both count their own thing, and on
this model both were satisfied by output that answers nothing (rung 5). Any
checkpoint this campaign ships has to be read by a judge that can say "this has
the shape of an answer and no answer in it". The steer-test judge lives in
score_with_judge.py and expects that file's schema, so this is the transcript
sibling: same labels, same label-order permutation, same disagreement rate.

Human prompts are the ones the gates ran (held-out split); benign prompts come
from DEFAULT_HARMLESS. Every response is judged twice with the label order
permuted; a label that flips is reported as DISAGREE, not averaged away.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_with_judge import LABELS_A, LABELS_B, MODEL, _call, _prompt  # noqa: E402


def classify(entries: list[dict], label: str) -> list[dict]:
    rows = []
    for i, e in enumerate(entries, 1):
        resp = e.get("response") or ""
        if not resp.strip():
            continue
        a = _call(_prompt(e["prompt"], resp, LABELS_A))
        b = _call(_prompt(e["prompt"], resp, LABELS_B))
        la, lb = a.get("label", "ERROR"), b.get("label", "ERROR")
        rows.append({"group": label, "prompt": e["prompt"], "label_a": la, "label_b": lb,
                     "agree": la == lb and la != "ERROR",
                     "why_a": a.get("why"), "why_b": b.get("why"),
                     "response_head": resp[:600], "response_len": len(resp)})
        if i % 5 == 0:
            print(f"  {label}: judged {i}/{len(entries)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gate", default="refusal", help="gate block to judge (default refusal)")
    args = ap.parse_args()

    data = json.loads(args.transcript.read_text())
    gens = data.get("generations") or []
    blocks = [g for g in gens if g.get("gate") == args.gate]
    if not blocks:
        print(f"no '{args.gate}' entries in {args.transcript} "
              f"(gates present: {sorted({g.get('gate') for g in gens})})")
        return 3

    # the split deterministic: harmful prompts are the ones the refusal gate ran;
    # a benign pool is not in the transcript, so everything here is one group.
    print(f"judge={MODEL} transcript={args.transcript.name} gate={args.gate} n={len(blocks)}")
    rows = classify(blocks, args.gate)

    out = args.out or args.transcript.with_suffix(".judge.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\nwrote {out}")

    per = defaultdict(Counter)
    for r in rows:
        per[r["group"]][r["label_a"] if r["agree"] else "DISAGREE"] += 1
    print("\ngroup | REFUSE | ANSWER | GARBLED | disagree | n")
    for g in sorted(per):
        c = per[g]
        print(f"{g:8s} | {c['REFUSE']:6d} | {c['ANSWER']:6d} | {c['GARBLED']:7d} | "
              f"{c['DISAGREE']:8d} | {sum(c.values())}")
    agree = sum(1 for r in rows if r["agree"])
    print(f"\nlabel-order agreement: {agree}/{len(rows)} = "
          f"{100.0 * agree / max(1, len(rows)):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
