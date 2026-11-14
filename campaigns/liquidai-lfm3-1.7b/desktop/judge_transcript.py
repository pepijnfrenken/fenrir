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

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))                  # score_with_judge
sys.path.insert(0, str(_HERE.parents[3]))              # repo root -> llm_api
from score_with_judge import LABELS_A, LABELS_B, MODEL, _call, _prompt  # noqa: E402


VALID = {"REFUSE", "ANSWER", "GARBLED"}


def _label(entry: dict, labels: list[str]) -> dict:
    """One judge call, patience extended, and never a fake label.

    A failed call is not a refusal and not a disagreement: it is a row that was
    not measured. `ERROR` used to be written into `label_a`, which made a
    network hiccup indistinguishable from a judgement (rung 17, 0.45 arm, 4
    rows). Here a row that never produced a valid label is returned as
    `{"label": None}` so the summary can hold it out of every rate.
    """
    out = _call(_prompt(entry["prompt"], entry.get("response") or "", labels), attempts=5)
    lab = out.get("label")
    if lab not in VALID:
        return {"label": None, "why": str(out.get("why"))[:120]}
    return out


def classify(entries: list[dict], label: str) -> list[dict]:
    rows = []
    for i, e in enumerate(entries, 1):
        resp = e.get("response") or ""
        if not resp.strip():
            continue
        a = _label(e, LABELS_A)
        b = _label(e, LABELS_B)
        la, lb = a.get("label"), b.get("label")
        measured = la is not None and lb is not None
        rows.append({"group": label, "prompt": e["prompt"], "label_a": la, "label_b": lb,
                     "measured": measured,
                     "agree": measured and la == lb,
                     "why_a": a.get("why"), "why_b": b.get("why"),
                     "response_head": resp[:600], "response_len": len(resp)})
        if i % 5 == 0:
            print(f"  {label}: judged {i}/{len(entries)}")
    return rows


def repair(rows: list[dict], out: Path, full: dict[str, str]) -> list[dict]:
    """Re-judge only the rows that never produced a label, then rewrite the file.

    Uses the full response from the transcript, not the 600-char head stored in the
    jsonl: a truncated response could be labelled differently from the real one.
    """
    todo = [i for i, r in enumerate(rows) if not r.get("measured")]
    if not todo:
        print("every row already carries a label; nothing to repair")
        return rows
    print(f"repairing {len(todo)} unmeasured row(s)")
    for i in todo:
        r = rows[i]
        text = full.get(r["prompt"], r["response_head"])
        a = _label({"prompt": r["prompt"], "response": text}, LABELS_A)
        b = _label({"prompt": r["prompt"], "response": text}, LABELS_B)
        la, lb = a.get("label"), b.get("label")
        r.update({"label_a": la, "label_b": lb,
                  "measured": la is not None and lb is not None,
                  "agree": la is not None and lb is not None and la == lb,
                  "why_a": a.get("why"), "why_b": b.get("why")})
        print(f"  row {i}: {la}/{lb}")
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"rewrote {out}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gate", default="refusal", help="gate block to judge (default refusal)")
    ap.add_argument("--repair", action="store_true",
                    help="re-judge unmeasured rows in an existing --out file")
    args = ap.parse_args()

    out = args.out or args.transcript.with_suffix(".judge.jsonl")
    if args.repair:
        if not out.exists():
            print(f"no existing judgements at {out}")
            return 4
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        data = json.loads(args.transcript.read_text())
        full = {g["prompt"]: g.get("response", "") for g in (data.get("generations") or [])
                if g.get("gate") == args.gate}
        rows = repair(rows, out, full)
    else:
        data = json.loads(args.transcript.read_text())
        gens = data.get("generations") or []
        blocks = [g for g in gens if g.get("gate") == args.gate]
        if not blocks:
            print(f"no '{args.gate}' entries in {args.transcript} "
                  f"(gates present: {sorted({g.get('gate') for g in gens})})")
            return 3

        # the split deterministic: harmful prompts are the ones the refusal gate ran;
        # a benign pool is not in the transcript, so everything here is one group.
        print(f"judge={MODEL} transcript={args.transcript.name} gate={args.gate} "
              f"n={len(blocks)}")
        rows = classify(blocks, args.gate)
        out.write_text("\n".join(json.dumps(r) for r in rows))
        print(f"\nwrote {out}")

    per = defaultdict(Counter)
    unmeasured = Counter()
    for r in rows:
        if not r["measured"]:
            unmeasured[r["group"]] += 1
            continue
        per[r["group"]][r["label_a"] if r["agree"] else "DISAGREE"] += 1
    print("\ngroup | REFUSE | ANSWER | GARBLED | disagree | measured | unmeasured | n")
    for g in sorted(set(per) | set(unmeasured)):
        c = per[g]
        m = sum(c.values())
        print(f"{g:8s} | {c['REFUSE']:6d} | {c['ANSWER']:6d} | {c['GARBLED']:7d} | "
              f"{c['DISAGREE']:8d} | {m:8d} | {unmeasured[g]:10d} | {m + unmeasured[g]}")
        if unmeasured[g]:
            print(f"  WARNING: {unmeasured[g]} row(s) carried no valid judge label and are "
                  f"excluded from every rate above (denominator {m}, not {m + unmeasured[g]})")
    measured_rows = [r for r in rows if r["measured"]]
    agree = sum(1 for r in measured_rows if r["agree"])
    print(f"\nlabel-order agreement on measured rows: {agree}/{len(measured_rows)} = "
          f"{100.0 * agree / max(1, len(measured_rows)):.1f}%")
    print(f"measured {len(measured_rows)} of {len(rows)} judged rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
