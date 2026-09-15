# Fun abliteration targets — scout notes (2026-09-14)

Context: pick the next campaign(s) for Fenrir. DIY-first rule applies; the
live landscape scan is **PENDING** (HF API fetch blocked by an approval gate
on the hub — re-run with approval or from the desktop).

## Already staged in Fenrir — configs written, never run

| Target | Config | Why it's fun | Feasibility |
|---|---|---|---|
| **FLUX.2-klein-4B text encoder** | `models/flux2-klein.yaml` + KB plan (`flux2-klein-abliteration-plan.md`) | Abliterate an *image model's* brain — 36-layer Qwen3.5 encoder, refusal spikes L32–34; verification = end-to-end image gen; first of its kind in the KB | ~8GB bf16; plan dots ~$0.02 on Molab (any ≥16GB GPU works) |
| **LiquidAI/LFM-8B-MoE** | `models/lfm-8b-moe.yaml` | `bias_vectors` + `svd` on a real production MoE — the method the MoE work has been building toward; LiquidAI lineage continuity (3 campaigns already on this family) | ~16GB bf16 → 24GB+ GPU |
| **deepreinforce-ai/Ornith-1.0-9B** | `models/ornith-9b.yaml` | Mystery model, `trust_remote_code`, hub push target already chosen | ~18GB bf16 → 24–40GB GPU |
| **barunlm-35m** | `models/barunlm-35m.yaml` | "How small can a refusal circuit be?" — toy-scale instrument demo | CPU-able; desktop as-is |
| **diffusiongemma** | `models/diffusiongemma.yaml` | Second diffusion target (parallel to flux2) | TBD |

## Lanes to scout (need the live HF scan)

1. **Guard models** — abliterate a safety classifier (current-gen Llama-Guard /
   Qwen-Guard / ShieldGemma class). "Who judges the judge" writeup; new lane.
2. **Multilingual refusal** — BgGPT (INSAIT), EuroLLM, Apertus (Swiss, fully open):
   does refusal live in the same place in non-English models?
3. **Dense Qwen3.5 (GatedDeltaNet family)** — same arch as Ternary-Bonsai-27B minus
   the MoE + ternary confounds → isolates the Bonsai post-mortem questions.
4. **Next-gen thinking model** — `style_refusal.py` is ready; keyword-blind lesson applies.
5. **VLM/multimodal** — stretch: harness needs image input paths.

## Next steps

- [ ] Run the Step-0 landscape scan (trending + fresh releases + existing
      abliterated variants per candidate) — awareness only, never a gate.
- [ ] Pino: what's being added to the desktop (RAM/GPU/models)? → sets the runnable tier.
- [ ] Stage configs + model downloads for the chosen targets.

### Guard lane — campaign ladder (added 2026-09-14)

1. **Verdict instrumentation**: `guard_mode` readout (verdict-token logit margin +
   label parse), `gate_flag_rate` (TPR on held-out harmful pairs), baseline sanity
   ("can the pristine guard even flag?" → INSTRUMENT SUSPECT if not).
2. **Guard abliteration**: steer-test causality → weight edit on the verdict axis.
   Open geometry question: is the guard's harm axis the SAME direction as its chat
   sibling's refusal axis (per-layer cosine)? Aligned = one shared safety axis;
   orthogonal = "flags" and "refuses" are separate circuits.
3. **Stealth question** (refusal gone AND output not flagged by a PRISTINE guard):
   measure the trade first — pristine refusal reads as "safe" to a guard (trivially),
   abliterated compliance gets flagged → naive ablation *increases* visibility.
   Then hypothesis space: (a) guard coverage gaps (language / encoding / style —
   testable with the same pairs), (b) guard edit (hides only from the edited guard),
   (c) shaped edit with a black-box guard in the optimization loop (compliance ∧
   ¬flag). Honest priors: content-harm is hard to hide from a strong guard
   in-distribution; wins, if any, live at coverage edges. Defensive read: this is
   guard-robustness measurement.
