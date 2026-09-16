# Paper idea — guard abliteration: the flag goes, the detector stays

Status: DRAFT PITCH (2026-09-16). Everything below is supported by committed
results in this campaign; nothing here is aspirational except where marked
"backlog". Companion artifacts: `../README.md` (full record), `../results/*.json`.

Working title options:
1. **"The Flag Goes, the Detector Stays: Weight Surgery on a Generative Safety Guard"**
2. **"Recalibrated, Not Ruined: Auditing Abliterated Guard Models"**
3. "Guard Abliteration Is a Threshold Attack: Mode-Asymmetric Damage and One-Parameter Recovery"

## Pitch (one paragraph)

Open-weight "abliterated" guard checkpoints already circulate on model hubs
(mradermacher/Qwen3Guard-*-heretic GGUFs, MLX builds) and weight surgery on
generators is a well-studied attack family — yet no published work measures what
an abliteration edit actually does to a *guard model's judgment*. We characterize
it from scratch on Qwen3Guard-Gen-0.6B across an edit-strength ladder (α =
1.0–1.5) and seven elicitation arms, on 80+80 held-out pairs. The edit decays the
guard's flag rate monotonically (28 → 10 → 2 → 0 of 80 harmful pairs on the read
it was trained against) while the verdict-token margins still rank harmful above
benign (AUC 0.982–0.955 across the calibration zone; 0.978 at the α=1.1 recipe) —
a recalibration, not information removal. The label rate understates recoverable
detection by up to ~30×: refitting a decision threshold on the ablated guard's
own margins restores 78/80 detections at α = 1.1 where the label says 10/80. The
footprint is **mode-asymmetric**, now as a precision cost: at matched recall
(≈78/80), recovery through the edited (response) read costs 15 benign false
positives; through the guard's untouched prompt-side read it costs **1**. The
untouched read also keeps the cleaner ordering (AUC 0.999 vs 0.978 at α = 1.1)
until both degrade at α = 1.5 (0.855 / 0.835) — the destruction line, where
recovery precision collapses (17–18 FPs). Flag-rate safety evaluations cannot
distinguish "recalibrated" from "gutted"; margin-level audits can, and the
detector is largely recoverable without touching weights.

## Abstract (draft, ~190 words — for the submission)

Safety guards are deployed as a last line of defense, and open-weight abliterated
guard checkpoints are already downloadable — but the effect of weight surgery on a
guard's judgment has never been characterized. We abliterate Qwen3Guard-Gen-0.6B
from scratch (layer-restricted projection of a harm direction, α ladder 1.0–1.5)
and measure it with a gated instrument on 80+80 held-out pairs: label parse plus
verdict-token margin, seven elicitation arms, both native moderation modes. The
flag rate decays monotonically with edit strength (28 → 10 → 2 → 0 of 80 on the
response read; 54 → 32 → 5 → 0 on the prompt read) — yet the margins keep ranking
harmful above benign (AUC 0.955–0.999 through α = 1.25), and re-thresholding
those margins recovers 78/80 detections out-of-sample at α = 1.1: the edit is a
recalibration, and the label read understates recoverable detection by up to
~30×. The damage is mode-asymmetric, visible as precision: recovery costs 15
benign false positives through the edited response read versus 1 through the
untouched prompt read, at matched recall. At α = 1.5 both reads lose ordering
(AUC 0.855 / 0.835) — the destruction line. We argue that flag-rate evaluations
of tampered guards are blind, and give a recalibration audit that separates
recoverable from damaged detection.

## Contributions

1. **First measured characterization of weight-surgery abliteration on a
   generative guard** (as far as our bounded literature scan found — see
   positioning). 7 elicitation arms × 4 edit strengths × 2 native reads, all on
   held-out pairs, all with margin-level records.
2. **Mode asymmetry — partially cross-family.** The edit subtracts from the read
   its direction was derived from (AUC at α=1.1: 0.978 edited read vs 0.999
   untouched); both reads degrade only at α=1.5 (0.855 / 0.835). **Flip test
   (qwen):** a prompt-derived direction excised at the same window leaves the
   response read nearly intact (72/80) while suppressing the prompt read (26/80);
   the response-derived direction hits both (10/80 · 32/80). **Cross-family
   (granite-guardian-3.1-2b, compared at matched own-read damage):** the
   response-derived edit's own-read selectivity **replicates** (6/80 own vs
   26/80 other at α=0.75); the prompt-derived edit does **not** (20/80 own vs
   8/80 other — anti-selective across the whole α ladder). Direction geometry is
   similar in both families (cos 0.87–0.94) — the footprint is not explained by
   alignment; the candidate explanation is scaffold overlap (granite's response
   scaffold embeds the user message). Across three independent manipulations the
   **response read is the fragile side** (all-modules ladder; granite
   prompt-derived edit; top-strip edits in both families).
3. **The residual is fragile — suppression is robust**: the prompt-side residual
   (slice2: 32/80) collapses to 3/80 with one benign context turn (pristine
   holds 80/80 under the same scaffold); a strictness system message re-elicits
   nothing extra on the killed read (8/80 vs pristine 78/80).
4. **Recalibration audit & one-parameter recovery**: the label read decays
   28 → 10 → 2 → 0/80 (edited read) and 54 → 32 → 5 → 0 (untouched read) while
   recoverable detection from the same margins holds at 78/78/67/65 and
   80/78/76/61 of 80 — at α=1.25 the label shows 2/80 and 67/80 are recoverable
   (33×). Flag-rate audits cannot distinguish recalibration from damage; a
   threshold refit (fit on tune, eval on test) recovers detection without
   touching weights, and its precision cost measures how far the edit went.
   Instantiates the ranking/calibration/threshold decomposition (cf. "Measuring
   the Wrong Thing") in the weight-surgery setting. **Threshold-free operating
   points** (TPR@5%FPR with 95% cluster-bootstrap CIs; `stats_ci.py`: 135 cells,
   158 matched-pair deltas): pristine 98.8 [96.2, 100] (resp) / 100 (prompt);
   α=1.1 96.2 [85.0, 100] / 100; α=1.25 82.5 [62.5, 92.5] / 95.0 [63.7, 100];
   α=1.5 28.7 [8.7, 51.2] / 45.0 [7.5, 60.0]. The α=1.0–1.1 collapse is a
   threshold shift, not information loss (AUC holds at 0.978–0.999; CIs overlap
   pristine); ranking damage starts at α≥1.25. Recovery replicates on the second
   family (granite rd/pd 0/80 → 76–80/80 at ≤5% FPR, AUC ≥0.983).
5. **Defender-side cost measurements (secondary)**: the visibility price of
   abliteration to a content-guard (+0.925 flip; margins −8.99 → +5.93 on the
   subject's responses); dilution evasion is order-dominated (harmful span first
   + benign flood → 0/10) and closed by sentence-level reads (10/10, zero false
   positives).
6. **A reproducible from-scratch pipeline** (pair builder, gates, probe, excise,
   readouts, tests) that runs on a single 8 GB card at ~zero cost.

## Evidence inventory (all committed under campaigns/guard-lane/)

| evidence | file |
|---|---|
| Visibility price (+0.925) | `results/visibility_price_lfm_pair_n40x2*.json` |
| Recipe replication (α ladder, gates) | `results/guard_gates_*.json`, `excise_repl_l17-18-19_a1*.json` |
| Elicitation matrix (7 arms × 2 guards) | `results/elicitation_probe_{pristine,a1.1}*.json` (slice2 = n=80+80) |
| Dilution + mitigations | `results/dilution_probe_*.json`, README "mitigation ladder" |
| Recalibration ladder (5 guards × 2 arms) | `results/recalibration_slice2-*.json` (+ pilot `recalibration_*.json`) |
| Fresh-slice replication (visibility price, pairs, generations) | `*_n120x2_off40*` files + `data/guard_pairs_lfm-abl_slice2_n120.jsonl` |
| Per-item raw records (margins, verdicts) | inside every JSON above + `items_*.jsonl` |
| Second family (granite-guardian-3.1-2b): acceptance, ladder, 2×2, recovery | `results/elicitation_probe_granite-*.json`, `results/recalibration_slice2-granite-*.json`, `results/directions_granite-{resp,prompt}.npz`, `models/granite-guardian-3.1-2b.yaml` |
| Bootstrap CIs + fixed-FPR operating points | `results/stats_ci.json` + `stats_ci.py` |

## Figures

- **F1 — Setup.** The generative guard's two native reads (prompt-moderation,
  response-moderation scaffolding) + pipeline diagram (pairs → probe → steer
  gate → excise → gates → probes).
- **F2 — The money figure.** x = α (pristine, 1.0, 1.1, 1.25, 1.5); flag rate
  (flat 0/10 from 1.0) vs recovered flags (9/10 · 9/10 · 9/10 · 3/10) vs AUC
  (0.96 · 0.955 · 0.895 · 0.69). Annotation: "same label, different damage".
- **F3 — Margin distributions.** Per-bank margin strips: pristine vs α=1.1
  (shifted, ordering intact) vs α=1.5 (interleaved = destruction).
- **F4 — Elicitation matrix.** 7 arms × {pristine, ablated} flag-rate heatmap;
  side panel: fragility (context) + instruction-proof (system-role) deltas +
  **read-specificity grid** (4 direction×window arms, both reads).
- **F5 — Dilution & mitigation.** Flag rate by order/pile size; windowed/sentence
  reads with false-positive controls.
- **(optional) F6 — Visibility price.** Subject-refusal flip as read by the guard.

## Positioning (bounded scan 2026-09-16; full related-work pass still needed)

*Generator-side abliteration (well covered — we are not that):* Arditi et al.
2024 (single refusal direction); Heretic tooling; defenses: extended-refusal
fine-tuning (2505.19056), refusal aliases (2608.18093), safety-pretraining
robustness (2510.02768).

*Guard models as an attack surface (covered, but input-side or training-side):*
mutations & adversarial attacks (Findings EMNLP 2025); controlled-release
prompting (USENIX Sec '26); Prompt Overflow (2605.23196); Super Suffixes
(2512.11783); refusal-cue shortcut (2608.03201 — closest neighbor: shows
shortcut reliance and legitimate refusal recognition are *partially separable*
via component masking — convergent with our mode asymmetry, different method);
guard confidence calibration (2410.10414); guard collapse under benign
fine-tuning (2605.02914 — different mechanism: SFT, geometry metrics).

*Audit methodology:* "Measuring the Wrong Thing" (2608.09624) — ranking vs
calibration vs threshold decomposition; we instantiate it for guard weight edits.
"Style over Substance" (2609.08236) — judges are style-gameable; our guard read
content not packaging (style edge negative), a useful contrast.

*The gap we occupy:* weight surgery on a guard, mode-resolved, with a recovery
audit. No paper found that does this; one earlier lane result (this campaign)
is the closest thing and it is ours.

*Practice already exists:* heretic GGUFs / MLX abliterated Qwen3Guard variants
are downloadable today; nobody has published what they do. "Here is what those
checkpoints actually do" is the hook.

## Threats to validity (and the plan)

1. **n**: slice-2 (fresh 80+80) landed 2026-09-16 and corrected the pilot's
   small-n flat-zero read into a monotone decay — flag reads are treacherous at
   n=10. Scale to ≥100 per bank and replicate across further slices (builder +
   subject generations; runs on the same card). *Top priority.*
2. **One guard family, one subject pair** → Granite Guardian (2B, has
   prompt+response modes) and/or Llama Guard 4 (8B) as second family.
3. **One edit family** (L-plurality, o_proj+down_proj, diff-of-means direction)
   → α ladder + lm_head negative control are within-family controls; add a
   per-layer sweep at 4B and a PCA-vs-diff-of-means ablation.
4. **Threshold-refit fairness**: fit uses the tune split (in-distribution for the
   direction extraction, never the eval split); report additionally an oracle
   threshold on test as the upper bound and the full recovered PR curve.
5. **Margin as instrument**: verdict-token logit margin, parse↔margin agreement
   160/160 on the pristine run; misses are boundary-adjacent (documented).
6. **Slot stability**: recipe replicated across 2 slices + cross-slice at 0.6B;
   elicitation/recalibration measured on one pair bank — replicate across the
   other slices.
7. **BG edge** (7–9/10, one robust miss) is translator-sensitive → gold human
   translations before any language claim.
8. **Scale**: everything here is 0.6B — the 4B runbook (README) is the
   generality test; do NOT copy the 0.6B layer window blindly.

## Open questions this would ship with

- ~~Does the mode asymmetry track the *data* the edit was derived from, or the
  architecture?~~ **ANSWERED (2026-09-16 flip test): the derivation read.** At
  matched depth (L17–19) a prompt-derived edit suppresses the prompt read and
  spares the response read; the response-derived edit hits both. Bonus mechanism
  datapoint: at L24–26 both directions are inert at α=1.1 despite higher
  separation — separation ≠ editability; and steering is read-agnostic at strong
  α while the response read needs ~2× the push from the prompt-derived direction.
- Is there a training-time defense that puts the two reads on one substrate
  (cf. the fine-tuning-collapse mitigation literature)?
- Can a "tamper audit" (margins + recovery) be run black-box, i.e. from queries
  alone, without weights?

## Venue & framing

Defensive framing throughout: guard-robustness measurement; the recipe is
community-standard practice and the characterization *aids defenders* (auditable
tampering + recovery without retraining). Target: safety/interpretability
workshop first (short paper), then decide on main-conference scale-up — that
decision is gated on the n-scale + second-family items above.

## Next actions (ordered)

1. **n-bump**: extend the pair bank to ≥100 harmful + ≥100 benign held-out
   pairs; rerun gates + elicitation + recalibration. (on-box, cheap)
2. **Slice replication** of elicitation + recalibration (other pair slices).
3. **Recovered PR curves** + oracle-threshold upper bound figure.
4. 4B/8B scale-up via the README runbook (Pino has compute).
5. Second guard family (Granite Guardian 2B) feasibility in the harness.
6. Full related-work pass before any submission.
