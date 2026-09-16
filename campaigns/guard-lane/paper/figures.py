#!/usr/bin/env python3
"""Guard-lane paper figures — generated from the committed result JSONs.

Figures (written to paper/figs/):
  fig_recalibration.png   — the money figure: label flags (flat 0/10) vs
                            recovered flags vs AUC, across the α ladder
  fig_margins.png         — per-item verdict margins (pristine / α=1.1 / α=1.5):
                            level shift with ordering intact vs destruction
  fig_pr_recovered.png    — recovered precision-recall curves per guard
                            (eval split) + fitted operating points

Usage:
  python campaigns/guard-lane/paper/figures.py \
      [--prefix slice2-] [--dir campaigns/guard-lane/results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
FIGS = HERE / "figs"

LADDER = [("pristine", "pristine"), ("a1.0", r"$\alpha$=1.0"), ("a1.1", r"$\alpha$=1.1"),
          ("a1.25", r"$\alpha$=1.25"), ("a1.5", r"$\alpha$=1.5")]


def load(prefix: str, rdir: Path) -> dict[str, dict]:
    out = {}
    for tag, _ in LADDER:
        p = rdir / f"recalibration_{prefix}{tag}.json"
        if p.exists():
            out[tag] = json.load(open(p))
    return out


def fig_recalibration(data: dict[str, dict], prefix: str) -> Path:
    tags = [t for t, _ in LADDER if t in data]
    labels = [lab for t, lab in LADDER if t in data]
    lab_h = [data[t]["arms"]["resp_mod"]["eval_label_only"]["label_flags_harmful"] for t in tags]
    rec_h = [data[t]["arms"]["resp_mod"]["eval"]["recal_flags_harmful"] for t in tags]
    rec_b = [data[t]["arms"]["resp_mod"]["eval"]["recal_flags_benign"] for t in tags]
    auc = [data[t]["arms"]["resp_mod"]["eval"]["auc"] for t in tags]
    n_h = data[tags[0]]["arms"]["resp_mod"]["eval"]["n_harmful"]
    uo_rec_h = [data[t]["arms"]["user_only"]["eval"]["recal_flags_harmful"] for t in tags]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(tags))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [h / n_h for h in lab_h], w, label="label read (as-is)", color="#d62728")
    ax.bar([i + w / 2 for i in x], [h / n_h for h in rec_h], w, label="recovered (threshold refit)", color="#2ca02c")
    ax.plot([i + w / 2 for i in x], [h / n_h for h in uo_rec_h], "o--", color="#1f77b4",
            label="recovered, prompt-side read")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"harmful pairs flagged (of {n_h})", color="black")
    ax.set_ylim(-0.02, 1.05)
    ax2 = ax.twinx()
    ax2.plot(list(x), auc, "s-", color="#444444", markersize=5, label="AUC (margin ranking)")
    ax2.set_ylabel("AUC", color="#444444")
    ax2.set_ylim(0.6, 1.02)
    ax2.tick_params(axis="y", colors="#444444")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8, framealpha=0.9)
    ax.set_title("Same '0/10' label, different damage: recovery across the edit ladder")
    for i, b in zip(x, rec_b):
        if b:
            ax.annotate(f"{b} FP", (i + w / 2, rec_h[i] / n_h), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7, color="#2ca02c")
    fig.tight_layout()
    out = FIGS / f"fig_recalibration{('_' + prefix.rstrip('-')) if prefix else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_margins(data: dict[str, dict], prefix: str, guards=("pristine", "a1.1", "a1.5")) -> Path:
    fig, axes = plt.subplots(1, len(guards), figsize=(10.5, 3.4), sharex=True)
    for ax, tag in zip(axes, guards):
        items = data[tag]["arms"]["resp_mod"]["eval_items"]
        for y, bank, color in ((0, "Unsafe", "#d62728"), (1, "Safe", "#1f77b4")):
            pts = [it["margin"] for it in items if it["expected"] == bank]
            ax.scatter(pts, [y] * len(pts), s=28, alpha=0.75, color=color)
        t = data[tag]["arms"]["resp_mod"]["threshold_fit"]["threshold"]
        ax.axvline(t, color="#2ca02c", lw=1.2, ls="--")
        ax.axvline(0.0, color="#999999", lw=1.0, ls=":")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["harmful", "benign"], fontsize=8)
        ax.set_ylim(-0.6, 1.6)
        lab = dict(LADDER)[tag]
        ax.set_title(f"{lab}\nAUC {data[tag]['arms']['resp_mod']['eval']['auc']}", fontsize=9)
        ax.set_xlabel("verdict margin")
    fig.suptitle("Verdict margins (response-mod read, eval split) — dashed green = refit threshold, dotted = label boundary",
                 fontsize=9)
    fig.tight_layout()
    out = FIGS / f"fig_margins{('_' + prefix.rstrip('-')) if prefix else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_pr(data: dict[str, dict], prefix: str, arm="resp_mod") -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    colors = {"pristine": "#444444", "a1.0": "#2ca02c", "a1.1": "#1f77b4", "a1.25": "#ff7f0e", "a1.5": "#d62728"}
    for tag, _ in LADDER:
        if tag not in data:
            continue
        items = data[tag]["arms"][arm]["eval_items"]
        ms = sorted({it["margin"] for it in items})
        thresholds = [ms[0] - 1] + [(ms[i] + ms[i + 1]) / 2 for i in range(len(ms) - 1)] + [ms[-1] + 1]
        rec, prec = [], []
        for t in thresholds:
            tp = sum(1 for it in items if it["expected"] == "Unsafe" and it["margin"] > t)
            fp = sum(1 for it in items if it["expected"] == "Safe" and it["margin"] > t)
            if tp + fp == 0:
                continue
            rec.append(tp / max(1, sum(1 for it in items if it["expected"] == "Unsafe")))
            prec.append(tp / (tp + fp))
        ax.plot(rec, prec, "-", color=colors.get(tag, "black"), lw=1.4, label=tag)
        # fitted operating point from tune
        ev = data[tag]["arms"][arm]["eval"]
        if "recal_flags_harmful" in ev:
            r = ev["recal_flags_harmful"] / max(1, ev["n_harmful"])
            p = ev["recal_flags_harmful"] / max(1, ev["recal_flags_harmful"] + ev["recal_flags_benign"])
            ax.scatter([r], [p], color=colors.get(tag, "black"), s=34, zorder=5)
    ax.set_xlabel("recall (harmful pairs)")
    ax.set_ylabel("precision")
    ax.set_title(f"Recovered precision-recall, {arm} read (eval split)\npoints = tune-fitted operating point")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    out = FIGS / f"fig_pr_recovered{('_' + prefix.rstrip('-')) if prefix else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="tag prefix, e.g. 'slice2-'")
    ap.add_argument("--dir", default=str(CAMPAIGN / "results"))
    args = ap.parse_args()
    data = load(args.prefix, Path(args.dir))
    if not data:
        raise SystemExit(f"no recalibration_*{args.prefix}*.json found in {args.dir}")
    for fn in (fig_recalibration, fig_margins, fig_pr):
        p = fn(data, args.prefix)
        print("wrote", p)


if __name__ == "__main__":
    main()
