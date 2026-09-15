#!/usr/bin/env python3
"""Build the guard-pair evaluation set — (prompt, response, expected verdict).

Pairs come from a subject's measured generations (the ``generations_*.jsonl``
files written by ``guard_visibility_price.py``): harmful-bank prompts with the
subject's response => expected ``Unsafe``; benign-bank prompts => expected
``Safe``. The train/tune/test split reuses ``eval_split.build_split`` so the
guard campaign's held-out set is as immutable as every chat campaign's.

Usage:
  python campaigns/guard-lane/build_guard_pairs.py \
      --gen-file campaigns/guard-lane/results/generations_lfm_pair_n40x2.jsonl \
      --subject ablated \
      --out campaigns/guard-lane/data/guard_pairs_lfm-abl_n40.jsonl \
      --train 20 --tune 10 --test 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from eval_split import build_split  # noqa: E402


def dig(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def load_generations(path: Path, subject: str) -> tuple[list[dict], list[dict]]:
    harmful, benign = [], []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if subject and rec.get("subject") != subject:
            continue
        (harmful if rec.get("bank") == "harmful" else benign).append(rec)
    return harmful, benign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-file", required=True)
    ap.add_argument("--subject", default="ablated",
                    help="subject whose responses become the pairs (default: ablated — "
                         "compliant harmful text is what a guard must flag)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--tune", type=int, default=10)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--seed", default="fenrir:guard:v1")
    args = ap.parse_args()

    harmful, benign = load_generations(Path(args.gen_file), args.subject)
    n = min(len(harmful), len(benign))
    if n == 0:
        sys.exit(f"no generations for subject={args.subject!r} in {args.gen_file}")
    harmful, benign = harmful[:n], benign[:n]
    total = args.train + args.tune + args.test
    if total != n:
        sys.exit(f"split sizes {args.train}+{args.tune}+{args.test}={total} != {n} pairs; "
                 f"adjust the sizes (n is set by the generations file)")

    split = build_split([r["prompt"] for r in harmful], [r["prompt"] for r in benign],
                        train_size=args.train, tune_size=args.tune, test_size=args.test,
                        seed=args.seed)
    # build_split partitions the HARMFUL list; the benign list is paired by
    # index (harmful[i] <-> benign[i]), so a benign prompt inherits the
    # partition of its paired harmful prompt.
    harmful_prompts = [r["prompt"] for r in harmful]
    part_of: dict[str, str] = {}
    for name, prompts in (("train", split.train), ("tune", split.tune), ("test", split.test)):
        for p in prompts:
            part_of[p] = name
            idx = harmful_prompts.index(p)
            part_of[benign[idx]["prompt"]] = name

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "tune": 0, "test": 0}
    with open(out_path, "w", encoding="utf-8") as f:
        for bank, expected, rows in (("harmful", "Unsafe", harmful), ("benign", "Safe", benign)):
            for r in rows:
                part = part_of.get(r["prompt"])
                if part is None:  # shouldn't happen: split covers every prompt
                    continue
                counts[part] += 1
                f.write(json.dumps({
                    "pair_id": dig(f"{bank}\0{r['prompt']}\0{r['response']}"),
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "expected": expected,
                    "bank": bank,
                    "subject": r.get("subject"),
                    "model": r.get("model"),
                    "split": part,
                }) + "\n")

    manifest = {
        "source_generations": str(args.gen_file),
        "subject": args.subject,
        "seed": args.seed,
        "sizes": counts,
        "total": sum(counts.values()),
        "split_ids": {"train": list(split.manifest["train_ids"]),
                      "tune": list(split.manifest["tune_ids"]),
                      "test": list(split.manifest["test_ids"])},
    }
    man_path = out_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path} ({sum(counts.values())} pairs: {counts})")
    print(f"wrote {man_path}")


if __name__ == "__main__":
    main()
