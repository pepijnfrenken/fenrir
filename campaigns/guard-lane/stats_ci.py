#!/usr/bin/env python3
"""Bootstrap CIs + fixed-FPR operating points for the guard-lane result files.

Handles both result schemas in ``results/``:

  - ``elicitation_probe_*.json``  → cells per (file, arm) from ``reads``
  - ``recalibration_*.json``      → per arm two cells: the model's own label rule
    (margin >= 0) and the refit operating point (margin >= t*, t* fitted on the
    *tune* split — a legitimate held-out threshold).

Per cell: flagH / flagB (95% cluster-bootstrap CIs over pair_ids), AUC
(Mann-Whitney), and TPR at fixed FPR 1% / 5% (threshold-free, benign margins
as negatives).  Paired deltas on matched pairs for:
  - ablated vs pristine (same family, same arm),
  - refit vs label (same file, same arm),
  - read vs read (resp_mod vs user_only, same file).

Pure numpy; no sklearn.  Usage:
  python stats_ci.py [--json out.json] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re

import numpy as np

B = 2000            # bootstrap resamples (TPR@FPR curves are O(n log n) each)
SEED = 20260916


def auc(h: np.ndarray, b: np.ndarray) -> float:
    """P(margin_harmful > margin_benign), ties at 0.5 (midrank formula)."""
    if len(h) == 0 or len(b) == 0:
        return float("nan")
    x = np.concatenate([h, b])
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    nh = len(h)
    return float((ranks[:nh].sum() - nh * (nh + 1) / 2.0) / (nh * len(b)))


def tpr_at_fpr(h: np.ndarray, b: np.ndarray, target: float) -> float:
    """Max TPR subject to FPR <= target."""
    if len(h) == 0 or len(b) == 0:
        return float("nan")
    ha, ba = np.sort(h), np.sort(b)
    cand = np.unique(np.concatenate([h, b]))[::-1]
    ch = len(h) - np.searchsorted(ha, cand, side="left")
    cb = len(b) - np.searchsorted(ba, cand, side="left")
    ok = (cb / len(b)) <= target
    return float((ch[ok] / len(h)).max()) if ok.any() else 0.0


def metrics(h: np.ndarray, b: np.ndarray, thr: float = 0.0) -> dict:
    return {
        "flagH": float((h >= thr).mean()) if len(h) else float("nan"),
        "flagB": float((b >= thr).mean()) if len(b) else float("nan"),
        "auc": auc(h, b),
        "tpr@1": tpr_at_fpr(h, b, 0.01),
        "tpr@5": tpr_at_fpr(h, b, 0.05),
    }


def boot(hid, h, bid, b, rng, thr: float = 0.0) -> dict[str, np.ndarray]:
    n_h, n_b = len(h), len(b)
    res = {k: np.empty(B) for k in ("flagH", "flagB", "auc", "tpr@1", "tpr@5")}
    for i in range(B):
        hi = rng.integers(0, n_h, n_h) if n_h else np.array([], int)
        bi = rng.integers(0, n_b, n_b) if n_b else np.array([], int)
        m = metrics(h[hi], b[bi], thr)
        for k in res:
            res[k][i] = m[k]
    return res


def ci(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load_file(path: str) -> dict[str, dict]:
    """-> {cell_name: {'arm','kind','thr','h':(ids,margins),'b':(ids,margins)}}"""
    d = json.load(open(path))
    cells: dict[str, dict] = {}
    if "reads" in d:                      # elicitation schema
        acc: dict[str, dict[str, list]] = {}
        for r in d["reads"]:
            arm, bank = r.get("arm"), r.get("bank")
            if arm is None or bank is None or r.get("margin") is None:
                continue
            acc.setdefault(arm, {"h": [], "b": []})
            acc[arm]["h" if bank == "harmful" else "b"].append((r["pair_id"], float(r["margin"])))
        for arm, c in acc.items():
            cells[arm] = {"arm": arm, "kind": "elicit", "thr": 0.0,
                          "h": _arr(c["h"]), "b": _arr(c["b"])}
    elif "arms" in d:                     # recalibration schema
        for arm, a in d["arms"].items():
            items = a.get("eval_items") or []
            rows = [(it["pair_id"], float(it["margin"]), it["expected"])
                    for it in items if it.get("margin") is not None]
            h = [(p, m) for p, m, e in rows if e == "Unsafe"]
            b = [(p, m) for p, m, e in rows if e == "Safe"]
            tstar = (a.get("threshold_fit") or {}).get("threshold", 0.0)
            cells[f"{arm}|label"] = {"arm": arm, "kind": "label", "thr": 0.0,
                                     "h": _arr(h), "b": _arr(b)}
            cells[f"{arm}|refit"] = {"arm": arm, "kind": "refit", "thr": float(tstar),
                                     "h": _arr(h), "b": _arr(b)}
    return cells


def _arr(rows: list) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.array([]), np.array([], dtype=float)
    return (np.array([p for p, _ in rows]),
            np.array([m for _, m in rows], dtype=float))


def aligned(hb: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, m = hb
    pos = {p: i for i, p in enumerate(ids)}
    k = [(j, pos[p]) for j, p in enumerate(idx) if p in pos]
    return (np.array([j for j, _ in k]), m[[i for _, i in k]])


def paired_delta(A: dict, B_: dict, key: str, rng) -> dict | None:
    hi = np.intersect1d(A["h"][0], B_["h"][0])
    bi = np.intersect1d(A["b"][0], B_["b"][0])
    if len(hi) < 5 or len(bi) < 5:
        return None
    jA_h, mA_h = aligned(A["h"], hi)
    jB_h, mB_h = aligned(B_["h"], hi)
    jA_b, mA_b = aligned(A["b"], bi)
    jB_b, mB_b = aligned(B_["b"], bi)
    n_h, n_b = len(hi), len(bi)
    point = metrics(mA_h, mA_b, A["thr"])[key] - metrics(mB_h, mB_b, B_["thr"])[key]
    d = np.empty(B)
    for i in range(B):
        hi_ = rng.integers(0, n_h, n_h)
        bi_ = rng.integers(0, n_b, n_b)
        va = metrics(mA_h[hi_], mA_b[bi_], A["thr"])[key]
        vb = metrics(mB_h[hi_], mB_b[bi_], B_["thr"])[key]
        d[i] = va - vb
    return {"delta": float(point), "ci": ci(d), "key": key, "n_h": n_h, "n_b": n_b}


def label_of(path: str) -> str:
    n = os.path.basename(path).replace(".json", "")
    return re.sub(r"^(elicitation_probe_|recalibration_)", "", n)


def baseline_for(label: str, avail: set[str]) -> str | None:
    if label.startswith("granite"):
        return "granite-pristine-slice2" if "granite-pristine-slice2" in avail else None
    if label in ("pristine-slice2", "pristine-arms2", "pristine"):
        return None
    if label.endswith("-slice2"):
        return "pristine-slice2"
    if label.endswith("-arms2"):
        return "pristine-arms2"
    return "pristine"


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--json", default=os.path.join(here, "results", "stats_ci.json"))
    args = ap.parse_args()

    files = sorted(globmod.glob(os.path.join(args.results, "elicitation_probe_*.json")) +
                   globmod.glob(os.path.join(args.results, "recalibration_*.json")))
    rng = np.random.default_rng(SEED)
    loaded = {label_of(f): load_file(f) for f in files}
    rows, deltas = [], {}

    for label, cells in sorted(loaded.items()):
        for cname, c in sorted(cells.items()):
            if len(c["h"][1]) == 0 or len(c["b"][1]) == 0:
                continue
            m = metrics(c["h"][1], c["b"][1], c["thr"])
            bt = boot(*c["h"], *c["b"], rng, c["thr"])
            row = {"cell": label, "arm": c["arm"], "kind": c["kind"], "thr": c["thr"],
                   "n_h": len(c["h"][1]), "n_b": len(c["b"][1]),
                   "flagH": m["flagH"], "flagH_ci": ci(bt["flagH"]),
                   "flagB": m["flagB"], "flagB_ci": ci(bt["flagB"]),
                   "auc": m["auc"], "auc_ci": ci(bt["auc"]),
                   "tpr@1": m["tpr@1"], "tpr@1_ci": ci(bt["tpr@1"]),
                   "tpr@5": m["tpr@5"], "tpr@5_ci": ci(bt["tpr@5"])}
            rows.append(row)
            print(f"{label:44s} {cname:18s} fH {m['flagH']*100:5.1f}% "
                  f"[{row['flagH_ci'][0]*100:5.1f},{row['flagH_ci'][1]*100:5.1f}] "
                  f"fB {m['flagB']*100:5.1f}% [{row['flagB_ci'][0]*100:4.1f},{row['flagB_ci'][1]*100:4.1f}] "
                  f"AUC {m['auc']:.3f}  TPR@1 {m['tpr@1']*100:5.1f}%  TPR@5 {m['tpr@5']*100:5.1f}% "
                  f"[{row['tpr@5_ci'][0]*100:5.1f},{row['tpr@5_ci'][1]*100:5.1f}]", flush=True)

    # ---- paired deltas ----
    for label, cells in sorted(loaded.items()):
        base = baseline_for(label, set(loaded))
        for key in ("tpr@5", "flagH"):
            if base and base in loaded:
                for cname, c in cells.items():
                    if c["kind"] != "elicit":
                        continue
                    bcell = loaded[base].get(cname)
                    if bcell:
                        dd = paired_delta(c, bcell, key, rng)
                        if dd:
                            deltas[f"{label}|{cname}|{key}|vs|{base}"] = dd
        # refit vs label (same file, same pairs): flagH is the refit story
        # (threshold moved); tpr@5 is included to document the invariance.
        for cname, c in cells.items():
            if "|refit" in cname:
                lcell = cells.get(cname.replace("|refit", "|label"))
                if lcell:
                    for key in ("flagH", "tpr@5"):
                        dd = paired_delta(c, lcell, key, rng)
                        if dd:
                            deltas[f"{label}|{c['arm']}|{key}|refit-vs-label"] = dd
        # read vs read (resp_mod vs user_only)
        arms = {c["arm"]: cn for cn, c in cells.items() if c["kind"] in ("elicit", "label")}
        if "resp_mod" in arms and "user_only" in arms:
            dd = paired_delta(cells[arms["user_only"]], cells[arms["resp_mod"]], "tpr@5", rng)
            if dd:
                deltas[f"{label}|tpr@5|user_only-vs-resp_mod"] = dd

    with open(args.json, "w") as fh:
        json.dump({"n_boot": B, "seed": SEED, "cells": rows, "deltas": deltas},
                  fh, indent=1, sort_keys=True)
    print(f"\nwrote {args.json}: {len(rows)} cells, {len(deltas)} paired deltas")


if __name__ == "__main__":
    main()
