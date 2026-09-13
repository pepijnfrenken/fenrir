---
campaign_id: <model-slug>-<date>
target_model: <org/model>
arch: <dense_llama | moe | ...> (<n> layers, hidden <h>, note non-square weights)
date: <YYYY-MM-DD>
landscape_scan: >
  AWARENESS-ONLY check (5 min, 2026-09-08): note published edits of the target/family
  (search HF: abliterated/uncensored/heretic; huihui-ai/insraq/mradermacher/sahilchachra;
  reproduce.json) for (a) honest framing of negative claims, (b) geometry hypotheses for
  OUR from-scratch configs. NEVER redirects the campaign: DIY from-scratch is primary;
  recovery forensics of a published edit is a LAST-RESORT rescue after the from-scratch
  map is exhausted (see campaigns/lfm2.5-recovery/).
status: <POSITIVE
status: <POSITIVE — gates passed | NEGATIVE — no config passes | PARTIAL — describe>
hardware: <Modal L4 | local CPU | ...>
cost: <approx $ / GPU-hrs>
# INSTRUMENT GATES (2026-09-13, ABSOLVER-1). Fill these BEFORE weight edits —
# both exist because minicpm5-2b green-lit three ablations on a baseline whose
# refusal the harness could not see (0/5 keyword-refused, refusal reasoning in
# every transcript; see campaigns/minicpm5-2b/README.md "Bugs found").
baseline_sanity: >
  `collect` on the PRISTINE model first: refusal_axis_measurable must be true
  (style-aware refusals found on the held-out set) and eval_pass's
  baseline_sanity gate green. instrument_suspect: true means the keyword
  readout is blind here — any keyword-derived refusal rate in this campaign is
  VOID, use bundle.refusal.style_refusals. A red baseline_sanity gate fails the
  run (exit 2) and NO ablation may be reported as passing.
pre_edit_steer_gate: >
  `harness/abl.py steer-test <config> --from-directions <bundle> --require-effect`
  BEFORE any weight edit (exit 3 = direction not causal). If steering cannot
  flip refusal at any alpha, projecting the direction into the weights cannot
  either (minicpm5-2b follow-up: 0/70 compliant over alpha -20..+20; the
  direction was a refusal-STYLE proxy). Record causal/effect_alphas.
methods_tried: [<advanced | mpoa | stacked_ablation | bias_vectors | lora | steering | direct_ablation>]
dir_methods_tried: [diff_means, paired, svd, leace, whitened_svd]
verdict_summary: >
  <2-3 sentence honest summary. If NEGATIVE: what was tried, what the
  blocker was, what the causal test showed. If POSITIVE: the winning
  config and gate numbers.>
key_numeric_results:
  pristine_refusal: <0-1>
  gate_refusal_ablated: <0-1>
  gate_coherence_ablated: <0-1>
  gate_capability_ablated: <0-1>
  best_config: {method: <>, dir_method: <>, alpha: <>, layers: <>, weights: <>}
bugs_found:
  - <bug_slug>   # one per bug: what silently corrupted results
recommended_next: [<bias_vector_residual_hook | larger_model | lora_finetune | ...>]
---

# Campaign: <Model Name>

## TL;DR
<2-3 sentences. Positive = the recipe. Negative = the blocker + evidence it's real.

## Why this model
<What made it a good target (size, arch, tuning, goal).>

## What happened (the honest arc)
<Numbered narrative: what was tried, in order, WITH the failures and false
starts. The false starts are the value — they're what the next campaign
won't repeat. Include: the config that looked good but wasn't, the causal
test that reframed it, the geometry/measurement insight.>

### N. <milestone heading>
<Per-milestone detail: method, config, result table, what it taught.>

## Bugs found & fixed (or still open)
| Bug | Mechanism | Consequence | Status |
|---|---|---|---|
| <name> | <mechanism> | <what it corrupted> | fixed `<commit>` / OPEN |

## What the NEXT campaign on this model should try first
<Ordered, with reasoning. The steering test result often dictates this.>

## Campaign flow with the instrument gates (do these in order)

1. `harness/abl.py inspect <config>` — arch, projection coverage, separation.
2. `harness/abl.py directions <config> [--dir-method paired]` — writes
   `directions-<flavor>-<dir_method>.pt` (flavor AND method are in the name; a
   paired harvest no longer overwrites the diff_means harvest).
3. `harness/abl.py collect <config> --transcript` on the PRISTINE model.
   **Gate:** `baseline_sanity` must be green (`refusal_axis_measurable: true`).
   Red = instrument suspect / unmeasurable axis ⇒ exit 2, STOP; no ablation
   result from this campaign is meaningful until the instrument sees refusal.
4. `harness/abl.py steer-test <config> --from-directions <the same .pt>
   --require-effect` on the candidate direction. **Gate:** exit 0 required —
   exit 3 means no alpha flipped refusal, so weight edits cannot either.
5. `harness/abl.py abl ...` then `collect --model-dir ... --transcript`. The
   ablated run is judged against the pristine bundle's refusal axis, so a
   green `eval_pass` requires the baseline to have been measurable.
6. Read the transcripts (not the counts) before writing the verdict.

## Key-numbers cheat-sheet
| Metric | Value |
|---|---|

---

## Hard-won rules (read before designing a config)

- **LFM2.5-hybrid: refusal spans ALL out-projections** (attn `out_proj` ×6,
  conv `out_proj` ×10, ffn `w2` ×16 = 32). Coverage > magnitude — huihui's
  published abliteration projects all 32 at rel_l2 0.022-0.030 and passes at
  default greedy; a 6-attention-only edit at 2× strength still refuses.
  `conv.out_proj` is a 2D hidden Linear and IS projectable — never
  blanket-exclude a projection class without a per-arch shape check.
- **Projection direction space is the OUTPUT space**: W[out, in] -= d(d^T W)
  needs d.shape[0] == out. ffn `w2` is [hidden, inter] (non-square on
  LFM2.5, measured [2048, 8192]) and still projects. Resolver weight names:
  `o_proj` (attn), `conv_out` (conv block, 2D-square gated), `w2`/`ffn_out`
  (ffn). Run `inspect` — it prints the resolved-coverage count.

## Template notes
- The YAML frontmatter is the MACHINE-READABLE part — keep every field
  filled so `harness/` scripts can mine correlations across campaigns.
- The narrative is the HUMAN-READABLE part — the false starts and the
  "why" are worth more than the final config.
- `status` drives the index: POSITIVE campaigns are the recipe source;
  NEGATIVE campaigns are the landmine map.
- Copy this file to `campaigns/<model-slug>/README.md` and fill it in as
  you go (not after — you'll forget the false starts).
