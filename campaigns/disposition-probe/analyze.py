#!/usr/bin/env python3
"""Analyse the disposition probe: per-arm mean stated probability + paired deltas.

All arms answer the SAME item grid, so the primary test is paired per item:
d_i = p_abliterated[i] - p_pristine[i]. Reported with a paired bootstrap CI
(B=2000, seed recorded) and a sign test that does not assume normality.
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

B = 2000
SEED = 20261113


def load(path: str) -> dict[str, dict[str, float]]:
    raw: dict[str, dict[str, list[float]]] = {}
    tones: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw.setdefault(r["arm"], {})
        # average repeated samples per item
        raw[r["arm"]].setdefault(r["item"], [])
        tones[r["item"]] = r.get("tone", "?")
        if isinstance(r.get("prob"), int):
            raw[r["arm"]][r["item"]].append(r["prob"])
    global TONE
    TONE = tones
    return {a: {i: sum(v) / len(v) for i, v in d.items()}
            for a, d in raw.items()}


TONE: dict[str, str] = {}


def boot_ci(xs: list[float], b: int = B) -> tuple[float, float]:
    rng = random.Random(SEED)
    n = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(b))
    return means[int(0.025 * b)], means[int(0.975 * b)]


def describe(name: str, xs: list[float]) -> None:
    m = st.mean(xs)
    lo, hi = boot_ci(xs)
    se = st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
    print(f"{name:>14}: n={len(xs):>3} mean={m:6.2f}  SE={se:5.2f}  "
          f"95%CI=[{lo:.2f}, {hi:.2f}]")


def main() -> None:
    path = sys.argv[1]
    arms = load(path)
    names = list(arms)
    base = "pristine" if "pristine" in arms else names[0]
    common = set.intersection(*(set(d) for d in arms.values()))
    print(f"arms={names}  base={base}  common items={len(common)}  file={path}\n")
    for a in names:
        describe(a, [arms[a][i] for i in sorted(common)])
    print("\nper tone (mean stated P(rise)):")
    for tone in ("bullish", "mixed", "bearish"):
        ids = sorted(i for i in common if TONE.get(i) == tone)
        if not ids:
            continue
        line = f"  {tone:>8} n={len(ids):>3} "
        for a in names:
            line += f"{a}={st.mean([arms[a][i] for i in ids]):5.2f}  "
        print(line)
    for a in names:
        if a == base:
            continue
        print()
        for tone in ("all", "bullish", "mixed", "bearish"):
            ids = sorted(i for i in common
                         if tone == "all" or TONE.get(i) == tone)
            d = [arms[a][i] - arms[base][i] for i in ids]
            if not d:
                continue
            lo, hi = boot_ci(d)
            pos = sum(x > 0 for x in d)
            neg = sum(x < 0 for x in d)
            flat = len(d) - pos - neg
            print(f"{a} - {base} [{tone:>7} n={len(d):>3}]: "
                  f"delta={st.mean(d):+6.2f} pp  95%CI=[{lo:+.2f}, {hi:+.2f}]  "
                  f"sign +{pos}/-{neg}/0{flat}  "
                  f"CI excludes 0: {lo > 0 or hi < 0}")


if __name__ == "__main__":
    main()
