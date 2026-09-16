#!/usr/bin/env python3
"""Guard lane — mitigation ladder: one table across every read mode.

Merges the dilution_probe outputs (full read / window 900 / window 350 /
sentence / resp_only / resp_only_sentences) into one per-condition table so the
recovery story reads as a ladder instead of six JSONs:

  flag count + margin mean per condition per mode, plus benign-control FPs.

Usage:
  python campaigns/guard-lane/mitigation_ladder.py
  python campaigns/guard-lane/mitigation_ladder.py --files full=dilution_probe_repl.json,sent=dilution_probe_repl-sent.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DEFAULT_FILES = [
    ("full", "dilution_probe_repl.json"),
    ("win900", "dilution_probe_repl-win.json"),
    ("win350", "dilution_probe_repl-win350.json"),
    ("sent", "dilution_probe_repl-sent.json"),
    ("respo", "dilution_probe_repl-respo.json"),
    ("respo+sent", "dilution_probe_repl-resposent.json"),
]


def load_modes(files) -> dict:
    data = {}
    for mode, fn in files:
        p = RESULTS / fn
        if not p.exists():
            print(f"# {mode:10s} {fn}: MISSING — skipped")
            continue
        d = json.load(open(p))
        meta = {k: d.get(k) for k in ("mode", "window_size", "window_stride", "min_len")}
        data[mode] = {"data": d, "meta": meta}
    return data


def cell(records, cond):
    flags, n, margins = 0, 0, []
    for rec in records:
        o = rec["conditions"].get(cond)
        if o is None:
            continue
        n += 1
        if o.get("band") == "flag":
            flags += 1
        if isinstance(o.get("margin"), (int, float)):
            margins.append(o["margin"])
    mm = sum(margins) / len(margins) if margins else float("nan")
    return flags, n, mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=None,
                    help="comma list of mode=filename overrides (default: the known ladder)")
    args = ap.parse_args()
    files = ([tuple(x.split("=", 1)) for x in args.files.split(",")]
             if args.files else DEFAULT_FILES)

    data = load_modes(files)
    if not data:
        print("no data files found")
        return

    print("modes:")
    for mode, d in data.items():
        print(f"  {mode:10s} {json.dumps(d['meta'])}  n_pairs={d['data'].get('n')}")

    conds: list[str] = []
    for d in data.values():
        for rec in d["data"]["per_pair"]:
            for c in rec["conditions"]:
                if c not in conds:
                    conds.append(c)

    hdr = f"{'condition':16s}" + "".join(f"{m:>14s}" for m in data)
    print()
    print(hdr)
    print("-" * len(hdr))
    for c in conds:
        row = f"{c:16s}"
        for mode, d in data.items():
            flags, n, mm = cell(d["data"]["per_pair"], c)
            row += f"{flags:>4}/{n:<2}{mm:+7.2f}" if n else f"{'—':>14s}"
        print(row)

    print()
    print("benign controls (false positives):")
    for mode, d in data.items():
        b = d["data"].get("benign_controls") or []
        fp = sum(1 for x in b if x.get("band") == "flag")
        margins = [x["margin"] for x in b if isinstance(x.get("margin"), (int, float))]
        mm = sum(margins) / len(margins) if margins else float("nan")
        print(f"  {mode:10s} {fp}/{len(b)} flagged  (margin mean {mm:+.2f})")


if __name__ == "__main__":
    main()
