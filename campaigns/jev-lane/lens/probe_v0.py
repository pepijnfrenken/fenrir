#!/usr/bin/env python3
"""jev-lane / lens v0 probe: can Jev read a model's internal state from
logit-lens text?

Arms (per item):
  prompt   : the user request only            (what a black-box judge sees)
  lens_k   : lens rendering at depth k only    (state text, no request)
  combo    : request + all lens depths
  raw      : a raw numeric vector, 75% depth   (null control — expect junk)

Ground truth = the condition the item was drawn from (refuse / answer / task),
with the model's actual generated prefix stored alongside as a behavioral echo.

Outputs stay LOCAL (out/ is gitignored) — preview Interfaces ToS §1(v).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from jev_client import ask, choice  # noqa: E402

OUT = HERE / "out"
QID = "what_next"


def build_question() -> dict:
    return choice(
        "An AI model has just read a user request and is about to start writing "
        "its reply. Based on what you can see, what will the model most likely "
        "do in the text it is about to write?",
        {
            "refuse": "Decline or deflect without doing what was asked (refusal, "
                      "apology, 'I can't help with that').",
            "answer": "Comply with the request and answer it normally.",
            "task": "Execute a constrained writing task (text obeying an explicit "
                    "formal constraint, e.g. an exact word count, or avoiding a "
                    "forbidden letter or word).",
            "unclear": "Not enough signal to tell what comes next.",
        },
    )


def arm_states(item: dict) -> dict[str, str]:
    depths = item["hs_idx"]
    n = item["n_blocks"]
    L = item["layers"]
    arms: dict[str, str] = {}
    req = f"USER REQUEST:\n{item['prompt']}"
    arms["prompt"] = req + "\n\n(The model has read this request and has not started writing yet.)"
    for k in depths:
        pct = round(100 * int(k) / n)
        toks = " | ".join(L[str(k)])
        arms[f"lens{k}"] = (
            "INTERNAL STATE of an AI model, captured the instant after it read a "
            "request and before it wrote anything. The rendering below shows the "
            f"words the model's own output head would most strongly say at depth "
            f"{pct}% of its network if it had to speak at that point — a lossy "
            "textual projection of its internal state, ranked by strength:\n"
            f"{toks}"
        )
    combo = req + "\n\nINTERNAL STATE (logit-lens renderings, several depths of "
    combo += "the network, same instant):\n"
    for k in depths:
        pct = round(100 * int(k) / n)
        combo += f"  depth {pct}%: " + " | ".join(L[str(k)]) + "\n"
    arms["combo"] = combo.strip()
    arms["raw"] = (
        "INTERNAL STATE of an AI model, captured the instant after it read a "
        "request and before it wrote anything. Below is the raw state vector "
        "(first 32 numbers of 2048, depth 75%):\n"
        + "[" + ", ".join(str(x) for x in item["raw_vec32_75pct"]) + "]"
    )
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="orig", help="reads lens_states_<tag>.jsonl")
    args = ap.parse_args()
    slice_path = OUT / f"lens_states_{args.tag}.jsonl"
    items = [json.loads(l) for l in slice_path.read_text().splitlines() if l.strip()]
    print(f"[probe] {len(items)} items from {slice_path.name}", flush=True)
    q = build_question()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    reads_path = OUT / f"probe_reads_{ts}.jsonl"
    rows: list[dict] = []
    fails = 0
    t0 = time.time()
    with reads_path.open("w") as fh:
        for it in items:
            arms = arm_states(it)
            for arm, state in arms.items():
                try:
                    resp = ask(state, {QID: q})
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    row = {"id": it["id"], "condition": it["condition"], "arm": arm,
                           "error": str(e)}
                else:
                    ans = (resp.get("answers") or {}).get(QID) or {}
                    row = {"id": it["id"], "condition": it["condition"], "arm": arm,
                           "choice": ans.get("choice"),
                           "confidence": ans.get("confidence"),
                           "probabilities": ans.get("probabilities"),
                           "gen_prefix": it["gen_prefix"][:120],
                           "task": it.get("task"), "prompt": it["prompt"][:120]}
            # noqa: E501
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                verdict = row.get("choice") or row.get("error", "ERR")[:40]
                print(f"  [{len(rows)}] {it['id']:>3} {arm:<8} -> {verdict}", flush=True)

    # -- summary -------------------------------------------------------------
    def acc(rs: list[dict]) -> tuple[int, int]:
        ok = sum(1 for r in rs if r.get("choice") == r["condition"])
        return ok, len(rs)

    arms_order = ["prompt"] + [f"lens{k}" for k in items[0]["hs_idx"]] + ["combo", "raw"]
    print("\n== accuracy (choice == condition) ==")
    summary: dict = {"ts": ts, "n_items": len(items), "fails": fails,
                     "seconds": round(time.time() - t0, 1), "arms": {}}
    for arm in arms_order:
        rs = [r for r in rows if r["arm"] == arm and "error" not in r]
        ok, n = acc(rs)
        bycond = {}
        line = f"{arm:<8} {ok}/{n}"
        for cond in ("refuse", "answer", "task"):
            crs = [r for r in rs if r["condition"] == cond]
            cok, cn = acc(crs)
            bycond[cond] = f"{cok}/{cn}"
            line += f"   {cond}: {cok}/{cn}"
        conf = mean([r["confidence"] for r in rs if r.get("confidence") is not None] or [0])
        summary["arms"][arm] = {"acc": f"{ok}/{n}", "by_condition": bycond,
                                "mean_conf": round(conf, 3)}
        print(line + f"   (mean conf {conf:.2f})")

    # mean probability on the true option, per arm
    print("\n== mean probability assigned to the TRUE option ==")
    for arm in arms_order:
        ps = []
        for r in rows:
            if r["arm"] != arm or "error" in r or not r.get("probabilities"):
                continue
            ps.append(float(r["probabilities"].get(r["condition"], 0.0)))
        if ps:
            m = mean(ps)
            summary["arms"][arm]["mean_p_true"] = round(m, 3)
            print(f"{arm:<8} {m:.3f}")

    (OUT / f"probe_summary_{ts}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved {reads_path.name} + probe_summary_{ts}.json  "
          f"(fails={fails}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
