# Guard lane — landscape scan (2026-09-15, live)

Awareness only (campaign rule): honest framing + later verification target; never
a gate, never routes the from-scratch work. Raw table:
`landscape-scan-raw-2026-09-15.txt` (HF API, 564 unique models seen across
`Qwen3Guard / Llama-Guard / ShieldGemma / Granite-Guardian / prompt-guard /
guardrail / guard / safety-classifier / harm-classifier`).

## Current-gen guard families (by downloads, 2026-09-15)

| Model | Params | License | Gated | Note |
|---|---|---|---|---|
| meta-llama/Prompt-Guard-86M | 86M | llama3.1 | manual | tiny prompt classifier; 4.5M dls |
| ibm-granite/granite-guardian-3.3-8b | 8B | apache-2.0 | no | enterprise taxonomy |
| ibm-granite/granite-guardian-4.1-8b | 8B | apache-2.0 | no | **2026-08-27** update |
| Qwen/Qwen3Guard-Gen-0.6B | 0.6B | apache-2.0 | no | generative verdicts, 119 langs → **our Step 2 target** |
| Qwen/Qwen3Guard-Gen-4B / 8B | 4B / 8B | apache-2.0 | no | scale-up rungs |
| Qwen/Qwen3Guard-Stream-0.6B | 0.6B | apache-2.0 | no | token-level head |
| meta-llama/Llama-Guard-3-1B/8B, Llama-Guard-4-12B | 1B–12B | llama | manual | S1–S13 taxonomy |
| google/shieldgemma-2b / 2-4b-it / 9b | 2B–9B | gemma | manual | prompt-harm classifier |
| nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 | 8B | other | no | |
| guardion/ModernGuard-1 | ? | ? | no | **2026-08-03** |
| astroware/Halo0.8B-guard-v1 | 0.8B | ? | no | **2026-08-26** |
| hfmlsoc/ncii-guard-v02 / ncii-light-guard-v01 | ? | ?/mit | no | **2026-08** |
| fastino/GLiNER2-Guardrails-PII-Multi | ? | apache-2.0 | no | PII-specific; 2026-08-11 |
| Intel/polite-guard | ? | apache-2.0 | no | politeness, not safety |

Also seen: "Guardpoint" 2026 tune family in GGUF mirrors (gemma-4-12B-it-Guardpoint,
Qwen3.5-27B-Guardpoint, Qwen3-14B-Guardpoint, gpt-oss-20b-Guardpoint) — worth a
card read when scaling the lane.

## Existing abliterated guard variants (AWARENESS ONLY)

- mradermacher/Qwen3Guard-Gen-8B-DeathGuard-heretic-i1-GGUF (~1.4k dls)
- mradermacher/Qwen3Guard-Gen-0.6B-heretic-i1-GGUF / -heretic-GGUF (~776/~235)
- rainmana/Qwen3Guard-Gen-0.6B-heretic (fp weights, ~13 dls)
- Otilde/Qwen3Guard-Gen-4B-Heretic (+MLX-Q8)

So: the whole Qwen3Guard-Gen family has "heretic" (community abliteration) variants —
the lane is not unexplored in the wild. Our differentiation is the instrumented,
from-scratch methodology (causal steer-test first, verdict-margin readout, geometry
question vs the chat sibling, visibility-price measurement) — not "first to do it".
Honest framing for any negative result must acknowledge these publics.
