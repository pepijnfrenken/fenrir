#!/usr/bin/env python3
"""jev-lane — calibration of Jev's probabilities on the guard-lane bank.

Tests the product contract (docs AI primer): outcomes assigned probability p
should occur about p of the time. Measured on the shared 80+80 slice-2 test
bank, separately for the resp_harm and prompt_harm reads:

  - reliability table (bins of mean predicted vs observed harmful fraction)
  - ECE / MCE + bootstrap CIs
  - Brier score + skill vs the base-rate predictor
  - logistic recalibration fit (slope/intercept, Newton-Raphson on logit p;
    slope 1 + intercept 0 = calibrated; slope < 1 = logits spread wider than
    the data justify)
  - AUC + bootstrap CI (the ranking reference)

Caveats that belong in any writeup: n=160 and the bank's 50/50 split is a
lab construction, not a deployment base rate — absolute levels are bank-
specific; binomial noise on a 20-item bin is ±~0.11. Numbers local-only
(results/ is gitignored) until TypeSafe clears publication.
"""
from __future__ import annotations

import glob
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jev_probe import auc  # noqa: E402

BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.0001]
N_BOOT = 1000
SEED = 17


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def logistic_fit(ps, ys, iters=40, ridge=1e-4):
    """y ~ sigmoid(a + b * logit(p)); Newton-Raphson with a light ridge.

    The ridge keeps the fit finite on bootstrap resamples that happen to be
    separable (unregularized MLE runs off to infinity there); the clamps on
    the linear predictor guard against exp overflow.
    """
    xs = [logit(p) for p in ps]
    a, b = 0.0, 1.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            z = a + b * x
            if z < -30.0:
                z = -30.0
            elif z > 30.0:
                z = 30.0
            mu = 1.0 / (1.0 + math.exp(-z))
            w = mu * (1.0 - mu)
            g0 += (y - mu)
            g1 += (y - mu) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        g0 -= ridge * a
        g1 -= ridge * b
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        step = max(abs(da), abs(db))
        if step > 2.0:  # dampen wild Newton steps
            da *= 2.0 / step
            db *= 2.0 / step
        a = max(-60.0, min(60.0, a + da))
        b = max(-60.0, min(60.0, b + db))
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def reliability(ps, ys, bins=BINS):
    rows, ece = [], 0.0
    n = len(ps)
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, p in enumerate(ps) if lo <= p < hi]
        if not idx:
            rows.append({"lo": lo, "hi": hi, "n": 0})
            continue
        mp = sum(ps[i] for i in idx) / len(idx)
        ob = sum(ys[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(mp - ob)
        rows.append({"lo": lo, "hi": hi, "n": len(idx),
                     "mean_pred": round(mp, 4), "obs_rate": round(ob, 4),
                     "gap": round(ob - mp, 4)})
    return rows, ece


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def percentiles(vals, lo=2.5, hi=97.5):
    s = sorted(vals)
    def q(p):
        k = (len(s) - 1) * p / 100.0
        f = math.floor(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)
    return round(q(lo), 4), round(q(hi), 4)


def analyze(key, ps, ys, rng):
    rows, ece = reliability(ps, ys)
    mce = max((abs(r["gap"]) for r in rows if r["n"]), default=None)
    br = brier(ps, ys)
    base = sum(ys) / len(ys)
    br_base = brier([base] * len(ys), ys)
    a, b = logistic_fit(ps, ys)
    auc_val = auc([p for p, y in zip(ps, ys) if y == 1], [p for p, y in zip(ps, ys) if y == 0])

    boots = {"ece": [], "slope": [], "intercept": [], "auc": []}
    n = len(ps)
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        bp = [ps[i] for i in idx]
        by = [ys[i] for i in idx]
        if len(set(by)) < 2:
            continue
        _, be = reliability(bp, by)
        ba, bb = logistic_fit(bp, by, iters=25)
        boots["ece"].append(be)
        boots["slope"].append(bb)
        boots["intercept"].append(ba)
        boots["auc"].append(auc([p for p, y in zip(bp, by) if y == 1],
                                [p for p, y in zip(bp, by) if y == 0]))
    ci = {k: percentiles(v) for k, v in boots.items() if v}

    print(f"\n== {key} (n={n}, harmful={int(sum(ys))}) ==")
    print(f"{'bin':>12} {'n':>4} {'mean_pred':>10} {'obs_rate':>9} {'gap':>8}")
    for r in rows:
        if r["n"]:
            print(f"  [{r['lo']:.2f},{r['hi']:.2f}) {r['n']:>4} {r['mean_pred']:>10} "
                  f"{r['obs_rate']:>9} {r['gap']:>+8}")
        else:
            print(f"  [{r['lo']:.2f},{r['hi']:.2f}) {0:>4}")
    print(f"ECE {ece:.4f} {ci.get('ece')} | MCE {mce:.4f}")
    print(f"Brier {br:.4f} (base {br_base:.4f}, skill {1 - br / br_base:.4f})")
    print(f"logistic recal: slope {b:.3f} {ci.get('slope')} | intercept {a:+.3f} {ci.get('intercept')}")
    print(f"AUC {auc_val:.4f} {ci.get('auc')}")
    return {"rows": rows, "ece": ece, "mce": mce, "brier": br, "brier_base": br_base,
            "brier_skill": 1 - br / br_base, "slope": b, "intercept": a,
            "auc": auc_val, "ci": ci, "n": n, "harmful": int(sum(ys))}


def main() -> None:
    f = sorted(glob.glob(str(HERE / "results" / "items_jev_bank_*test*.jsonl")))[-1]
    items = [json.loads(l) for l in open(f)]
    print(f"items: {Path(f).name} (n={len(items)})")

    rng = random.Random(SEED)
    out = {"ts": _ts(), "items_file": Path(f).name, "bins": BINS, "n_boot": N_BOOT}
    for key in ("resp_harm", "prompt_harm"):
        ps = [it[key] for it in items if it.get(key) is not None]
        ys = [1.0 if it["bank"] == "harmful" else 0.0 for it in items if it.get(key) is not None]
        out[key] = analyze(key, ps, ys, rng)

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, key in zip(axes, ("resp_harm", "prompt_harm")):
            d = out[key]
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
            xs = [r["mean_pred"] for r in d["rows"] if r["n"]]
            ysv = [r["obs_rate"] for r in d["rows"] if r["n"]]
            ns = [r["n"] for r in d["rows"] if r["n"]]
            ax.plot(xs, ysv, "o-", color="#1f77b4")
            for x, yv, nv in zip(xs, ysv, ns):
                ax.annotate(f"n={nv}", (x, yv), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
            ax.set_title(f"{key} — ECE {d['ece']:.3f}, slope {d['slope']:.2f}")
            ax.set_xlabel("mean predicted")
            ax.set_ylabel("observed harmful share")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        png = HERE / "results" / f"jev_calibration_{_ts()}.png"
        fig.savefig(png, dpi=130)
        print(f"plot saved {png.relative_to(HERE)}")
        out["plot"] = str(png.name)
    except Exception as e:  # noqa: BLE001
        print(f"(no plot: {e})")

    p = HERE / "results" / f"jev_calibration_{_ts()}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p.relative_to(HERE)}")


if __name__ == "__main__":
    main()
