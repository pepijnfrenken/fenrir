"""2x2 flip-test grid: {response-derived, prompt-derived direction} x {L17-19, L24-26}.

Prints, per arm: label flags on both reads (slice-2 test), recovered flags + FPs
(threshold refit on tune), AUC — plus the pristine row for reference.
Run from campaigns/guard-lane/.
"""
import json
from pathlib import Path

R = Path("results")

ARMS = [
    ("resp-derived @17-19 (existing a1.1)", "a1.1", "slice2-a1.1"),
    ("prompt-derived @17-19 (A)", "pd17-19-a1.1", "slice2-pd17-19-a1.1"),
    ("prompt-derived @24-26 (B)", "pd24-26-a1.1", "slice2-pd24-26-a1.1"),
    ("resp-derived @24-26 (C)", "rd24-26-a1.1", "slice2-rd24-26-a1.1"),
]

hdr = f"{'arm':38s} | {'resp_mod: label -> recovered (FP) AUC':43s} | {'user_only: label -> recovered (FP) AUC':43s}"
print(hdr)
print("-" * len(hdr))

def row(tag_el, tag_rec):
    el = R / f"elicitation_probe_{tag_el}.json"
    rc = R / f"recalibration_{tag_rec}.json"
    out = {}
    if rc.exists():
        d = json.load(open(rc))
        for arm in ("resp_mod", "user_only"):
            b = d["arms"][arm]
            e, f = b["eval"], b["threshold_fit"]
            out[arm] = (e["label_flags_harmful"], e["recal_flags_harmful"],
                        e["recal_flags_benign"], e["auc"], e["n_harmful"])
    elif el.exists():
        d = json.load(open(el))
        for arm in ("resp_mod", "user_only"):
            s = d["summary"].get(arm)
            out[arm] = (s["flagged_harmful"], None, None, s["auc"], s["n_harmful"])
    return out

# pristine reference
p = json.load(open(R / "recalibration_slice2-pristine.json"))
for arm in ("resp_mod", "user_only"):
    e = p["arms"][arm]["eval"]
    print(f"{'PRISTINE (reference)':38s} | {arm}: {e['label_flags_harmful']}/{e['n_harmful']} -> "
          f"{e['recal_flags_harmful']}/{e['n_harmful']} ({e['recal_flags_benign']}) AUC {e['auc']}")

for name, tag_el, tag_rec in ARMS:
    r = row(tag_el, tag_rec)
    parts = []
    for arm in ("resp_mod", "user_only"):
        v = r.get(arm)
        if not v:
            parts.append(f"{arm}: (missing)")
            continue
        lab, rec, fp, auc, n = v
        recs = f"-> {rec}/{n} ({fp})" if rec is not None else "-> (not run)"
        parts.append(f"{arm}: {lab}/{n} {recs} AUC {auc}")
    print(f"{name:38s} | " + " | ".join(parts))
