#!/usr/bin/env python3
"""Re-judge every stored gate transcript under ONE judge model.

Why: the judge backend was swapped (free one-slot `deepseek-v4-flash` -> commandcode
`deepseek/deepseek-v4.1-flash-fast`). Rows judged by the old model and rows judged by
the new one are not the same instrument, so cross-arm comparison only means something
when every row is re-read by the same backend. This script does that in parallel
(commandcode is multi-slot; the free endpoint was not) and writes a per-arm ledger of
old-vs-new label drift so the cost of the swap is measurable rather than assumed.

Outputs
  campaigns/liquidai-lfm3-1.7b/judge-cc-<run>-<arm>.jsonl   new labels
  campaigns/liquidai-lfm3-1.7b/judge-cc-ledger.json         per-arm counts + drift

Usage
  python desktop/rejudge_all.py                # everything under .forensics/
  python desktop/rejudge_all.py --runs r17 r20 # a subset
  python desktop/rejudge_all.py --workers 4 --redo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = HERE.parents[2]
FORENSICS = REPO / ".forensics"
JUDGE = HERE / "judge_transcript.py"


def transcripts() -> list[tuple[str, str, Path]]:
    out = []
    for run_dir in sorted(FORENSICS.glob("r*")):
        if not run_dir.is_dir():
            continue
        for f in sorted(run_dir.glob("hookgate-*.json")):
            arm = f.stem.replace("hookgate-", "")
            out.append((run_dir.name, arm, f))
    return out


def n_generations(path: Path) -> int:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    gens = d.get("generations") or []
    return len(gens) if isinstance(gens, list) else 0


def complete(out: Path, expect: int) -> bool:
    """A row-per-generation file with no unmeasured row for this arm."""
    if not out.exists() or expect == 0:
        return False
    try:
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    ok = [r for r in rows if r.get("measured") is not False and r.get("label_a")]
    return len(rows) >= expect and len(ok) >= expect


def counts(path: Path) -> dict:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return {}
    lab = [r.get("label_a") for r in rows if r.get("measured") is not False]
    c = {k: lab.count(k) for k in ("REFUSE", "ANSWER", "GARBLED")}
    c["unmeasured"] = len(rows) - len(lab)
    c["disagree"] = sum(1 for r in rows if r.get("measured") is not False and not r.get("agree"))
    c["n"] = len(rows)
    return c


def judge(run: str, arm: str, src: Path, redo: bool) -> dict:
    out = CAMPAIGN / f"judge-cc-{run}-{arm}.jsonl"
    expect = n_generations(src)
    if not redo and complete(out, expect):
        return {"run": run, "arm": arm, "state": "cached", **counts(out)}
    cmd = [sys.executable, str(JUDGE), str(src), "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        return {"run": run, "arm": arm, "state": f"FAIL rc={p.returncode}",
                "tail": (p.stderr or p.stdout)[-300:]}
    c = counts(out)
    c.update(run=run, arm=arm, state="ok" if c.get("unmeasured", 1) == 0 else "unmeasured")
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    jobs = [t for t in transcripts() if not args.runs or t[0] in args.runs]
    if not jobs:
        print("no transcripts", file=sys.stderr)
        return 1
    print(f"{len(jobs)} arms, {args.workers} workers, judge model from score_with_judge")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge, r, a, s, args.redo): (r, a) for r, a, s in jobs}
        for fut in futs:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                r, a = futs[fut]
                results.append({"run": r, "arm": a, "state": f"FAIL {type(exc).__name__}"})
    results.sort(key=lambda d: (d.get("run", ""), d.get("arm", "")))

    print(f"\n{'run':5} {'arm':34} {'REF':>4} {'ANS':>4} {'GAR':>4} {'dis':>4} {'unm':>4} {'n':>3}  state")
    for d in results:
        print(f"{d.get('run',''):5} {d.get('arm',''):34} {d.get('REFUSE',0):4} "
              f"{d.get('ANSWER',0):4} {d.get('GARBLED',0):4} {d.get('disagree',0):4} "
              f"{d.get('unmeasured',0):4} {d.get('n',0):3}  {d.get('state','')}")

    # drift against the old judge, where an old file exists
    drift = []
    for d in results:
        old = CAMPAIGN / f"judge-{d['run']}-{d['arm']}.jsonl"
        new = CAMPAIGN / f"judge-cc-{d['run']}-{d['arm']}.jsonl"
        if not old.exists() or not new.exists():
            continue
        o, nw = counts(old), counts(new)
        drift.append({"run": d["run"], "arm": d["arm"], "old": o, "new": nw})
    if drift:
        print(f"\n{'arm':40} {'old R/A/G':>12} {'new R/A/G':>12}")
        for d in drift:
            o, nw = d["old"], d["new"]
            print(f"{d['run'] + '/' + d['arm']:40} "
                  f"{o.get('REFUSE',0):4}/{o.get('ANSWER',0)}/{o.get('GARBLED',0):<4}"
                  f" {nw.get('REFUSE',0):4}/{nw.get('ANSWER',0)}/{nw.get('GARBLED',0):<4}")

    ledger = CAMPAIGN / "judge-cc-ledger.json"
    ledger.write_text(json.dumps({"results": results, "drift": drift}, indent=1) + "\n")
    print(f"\nwrote {ledger.relative_to(REPO)}")
    bad = [d for d in results if d.get("state", "").startswith("FAIL") or d.get("unmeasured")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
