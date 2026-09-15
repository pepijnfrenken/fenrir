---
campaign_id: guard-lane-2026-09-15
target_model: Qwen/Qwen3Guard-Gen-0.6B
arch: qwen3 dense (28 layers, hidden 1024, 16 heads / 8 KV, tied embeddings; base Qwen3-0.6B)
date: 2026-09-15
landscape_scan: >
  AWARENESS-ONLY (live scan, 2026-09-15 — full table in
  landscape-scan-raw-2026-09-15.txt). Guard families found: Qwen3Guard-Gen
  (0.6B/4B/8B, apache-2.0, ungated, 119 languages), Qwen3Guard-Stream-0.6B,
  granite-guardian 3.x/4.1 (apache-2.0, ungated; 4.1-8b updated 2026-08-27),
  Llama-Guard 3/4 (gated), ShieldGemma (gated), nvidia Nemotron Safety
  Guard 8B v3, plus small 2026 guards (ModernGuard-1, Halo0.8B-guard-v1,
  ncii-guard-v02, GLiNER2-Guardrails-PII). Existing abliterated guard
  variants ALREADY EXIST — mradermacher/Qwen3Guard-Gen-{0.6B,4B,8B}-heretic /
  DeathGuard-heretic (GGUF), rainmana/Qwen3Guard-Gen-0.6B-heretic (fp),
  Otilde/Qwen3Guard-Gen-4B-Heretic (MLX). Awareness only: for honest framing
  of negative claims and as a later verification target; NEVER redirects this
  campaign. From-scratch DIY remains primary.
status: IN-PROGRESS — Step 0 (visibility price) staged; full ladder in docs/guard-lane-spec.md
hardware: desktop WSL2 — RTX 3060 Ti 8 GB (subjects 1.2B pair + guard 0.6B, bf16)
cost: ~0 (local)
baseline_sanity: >
  Guard-side analog of the refusal baseline gate: the PRISTINE guard must
  measurably flag held-out harmful pairs (flag rate high) and NOT flag benign
  pairs, or the run is INSTRUMENT SUSPECT (a guard that reads everything Safe
  is the "keyword gate" mistake wearing a different hat). Step 0 smoke does
  this before any verdict table is trusted.
pre_edit_steer_gate: >
  Same discipline as chat campaigns: before any guard weight edit, verify the
  candidate direction is causal (steer-test flips the verdict). No causal flip
  → the direction is a content proxy, not the decision circuit.
methods_tried: []
dir_methods_tried: []
verdict_summary: >
  TBD. Step 0 = measure the visibility price (how guard verdicts react to
  abliterated subjects) before touching guard weights.
key_numeric_results:
  pristine_refusal: TBD
  gate_refusal_ablated: TBD
  gate_coherence_ablated: TBD
  gate_capability_ablated: TBD
  best_config: TBD
bugs_found: []
recommended_next: [visibility_price_run, guard_mode_instrumentation, steer_test]
---

# Campaign: Guard lane — abliterate the judge (Qwen3Guard-Gen-0.6B)

## TL;DR
New lane: the abliteration target is a **safety classifier** — the behavior under
study is *flagging* (verdict emission), not refusal. Ladder: **Step 0** measure the
visibility price (what a guard does to pristine vs abliterated subjects) → **Step 1**
guard-mode instrumentation (`guard_readout` + gates) → **Step 2** abliterate the
guard with the standard instrument gates → **Step 3** (stretch) stealth shaping /
coverage-edge measurement. Live guard landscape scan done (awareness only).

## Why this model
- Smallest current-gen *generative* guard (0.6B, apache-2.0, ungated) → runs
  anywhere; first rung of the ladder; 4B/8B siblings available for scale-up.
- Generative verdicts = a readable output to instrument: verdict-token margin +
  label parse, exactly parallel to the refusal readouts the harness already has.
- The geometric question is real and cheap here: is the guard's *harm* axis the
  same direction as its chat sibling's *refusal* axis (per-layer cosine)?
  Aligned = one shared safety axis; orthogonal = "flags" vs "refuses" are
  separate circuits. Either answer is a finding.

## What happened (the honest arc)
0. Scout (2026-09-14): lane identified; spec written (`docs/guard-lane-spec.md`);
   implementation ladder added to `docs/target-scout-2026-09-14.md`.
1. Landscape scan (2026-09-15, desktop up): guard families + published guard
   abliterations recorded above. `Qwen3Guard-Gen-0.6B` downloaded to desktop.
2. Step 0 staged (2026-09-15): `guard_visibility_price.py` + prompt banks from
   `data/refusal_eval.jsonl` (200 harmful / 200 benign, campaign-consistent with
   minicpm5-2b and if-ablation). Subjects: the published LFM2.5-1.2B pair
   (pristine vs abliterated) — the cheapest clean before/after we own.
   — result: see `results/` after the run.

### Step 0 — the visibility price (in flight)
Protocol: same prompts, greedy chat generations from both subjects; each
(prompt, response) judged by the guard with (a) label parse and (b) verdict-token
margin (`Unsafe` − `Safe` logits at the decision step). Report per-subject flag
rates on harmful + benign banks; the **visibility price** = flag-rate delta
between abliterated and pristine on the harmful bank. Expectation: pristine
refusals read as Safe (trivially), abliterated compliance gets flagged → naive
abliteration *increases* visibility. If so, Step 3 is where that gets attacked.

## Bugs found & fixed (or still open)
| Bug | Mechanism | Consequence | Status |
|---|---|---|---|
| — | — | — | — |

## What the NEXT campaign on this model should try first

1. Step 1: land `guard_readout.py` + `gate_flag_rate` / `gate_pass_rate_benign`
   onto the harness (spec in `docs/guard-lane-spec.md`), including the
   INSTRUMENT SUSPECT check ("can the pristine guard even flag?").
2. Step 2: probe → distill → steer-test → excise on the verdict axis, with the
   same instrument gates as every chat campaign.
3. Geometry read: per-layer cosine between the guard's harm axis and its chat
   sibling's refusal axis (Qwen3-0.6B base exists as a natural sibling).

## Campaign flow with the instrument gates (do these in order)

1. `guard_visibility_price.py --smoke` — guard sanity (flags canned harmful
   responses, parses labels, margins in the right direction).
2. `guard_visibility_price.py --n 40` — visibility price table (both subjects,
   both banks). This IS the Step-0 deliverable.
3. Step 1 instrumentation, then the standard arc: inspect → directions →
   collect (baseline sanity) → steer-test (require-effect) → abl → gates →
   read transcripts before writing the verdict.

## Key-numbers cheat-sheet
| Metric | Value |
|---|---|
| (filled by the Step-0 run) | — |

---

## Lane notes (read before designing a config)

- **Instrument-first inversion**: for chat targets the failure mode was a gate
  blind to refusals; for a guard target the analogous failure is a guard that
  flaps polarity (reads refusal as flag). The margin readout exists precisely to
  catch sign flips that a label parse could hide.
- **A guard edit hides the subject from THAT guard only** (Step 3 honest prior).
  Wins, if any, live at coverage edges (language / encoding / style) — and
  Bulgarian coverage is available for testing (Qwen3Guard claims 119 languages).
- **Defensive framing** for any writeup: this is guard-robustness measurement —
  same discipline as the chat campaigns (break it, document it, publish numbers).
