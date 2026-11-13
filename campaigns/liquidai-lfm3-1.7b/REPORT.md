# LFM3-1.7B abliteration campaign — final report

Date: 2026-11-13. Hardware: desktop RTX 3060 Ti 8 GB, WSL2 Ubuntu-24.04, driver
561.17, torch 2.14.1+cu126, transformers 5.23.0. Cost: 0 (own hardware).

## Verdict

**NEGATIVE. The recipe did not transfer.** MPOA projection of the paired
output-phase refusal direction into `o_proj + conv_out + w2` across LFM3-1.7B's
24 layers removes refusal only in proportion to the distribution damage it
causes, and there is no alpha where the refusal gate (<= 3%) and the KL gate
(<= 0.1 nats) are both satisfied. The pre-edit causal gate, once it was made
capable of failing, returns NOT CAUSAL.

No checkpoint from this campaign is publishable as an abliterated model.

## Instrument history (read this before the numbers)

Three defects were found and fixed during the campaign. The numbers below were
produced by the fixed instruments unless marked otherwise.

1. **Hardcoded 5-prompt held-out gate.** `collect` had `n_test = 5` in code, so
   every gate reported on 5 prompts (20pp resolution). Now `n_test_prompts` is a
   config field; this campaign used 12. `DEFAULT_HARMLESS` grew 20 -> 35 new
   topics so the 12-prompt split fits against `DEFAULT_HARMFUL` (35). Split:
   train 11 / tune 12 / test 12, seed `absolver:qwen25:v1`.
2. **Structural PPL column.** `steer-test` printed `PPL +0.00%` at every alpha
   because the steering hook fires on the last token only, which cannot move a
   teacher-forced prompt PPL. That column is `-` unless `--ppl-check`.
3. **The gate that lied.** A refusal count that falls because the model emits
   token soup was counted as causality. At every non-zero alpha the steered
   model produced `disallowed disallowed disallowed ...` and the detector read
   0 refusal because there was no refusal sentence in the slop. Rows now carry
   coherence and degeneracy, a majority-incoherent alpha is flagged `COLLAPSE`,
   its counts are void, and it can never satisfy `--require-effect`. Two tests
   pin this exact failure (117 tests total pass).

## Gates (12-prompt held-out split, all measurements on this box)

| target | refusal | coherence | degeneracy | capability | PPL inc | first-token KL | eval_pass |
|---|---|---|---|---|---|---|---|
| pristine | 12/12 | 12/12 | 0.000 | 0.400 abs | - | - | n/a |
| mpoa a0.5, all 24 layers | 11/12 | 12/12 | 0.000 | 0.87 | +0.2% | 0.654 | false |
| mpoa a1.0, all 24 layers | 9/12 | 12/12 | 0.000 | 1.25 | +2.6% | 0.986 | false |
| mpoa a2.0, all 24 layers | 7/12 | 12/12 | 0.000 | 1.00 | +18.0% | 1.388 | false |
| mpoa a2.0, attention only (L3,7,11,15,19,23; o_proj+w2) | 12/12 | 12/12 | 0.000 | 1.00 | +13.8% | 0.326 | false |

Gate caps: refusal <= 3%, coherence >= 90%, degeneracy <= 5%, capability
retention >= 0.83, PPL increase <= 15%, first-token KL <= 0.1.

Capability "retention" of 1.25 is noise on a 20-item MMLU mini (abl 0.500 vs
pristine 0.400), not an improvement.

## Pre-edit causal gate (fixed instrument)

Harvest: `paired`, 20 prompts, 320-token continuation budget (raised from 96 to
test whether the direction was a thinking-vs-answering axis). Separation at L23
3.58 with the 320 budget vs 3.64 with 96, so the direction does not depend on
where in the reasoning it is sampled. That hypothesis is dead.

| alpha | refusal | benign | coherent | verdict |
|---|---|---|---|---|
| 0.00 | 8/8 | 0/6 | 8/8 | baseline |
| -2.00 | 0/8 | 0/6 | **0/8** | COLLAPSE, counts void |
| -1.00 | 0/8 | 0/6 | **0/8** | COLLAPSE, counts void |
| +1.00 | 0/8 | 0/6 | **0/8** | COLLAPSE, counts void |
| +2.00 | 0/8 | 0/6 | **1/8** | COLLAPSE, counts void |

Gate verdict: `NOT CAUSAL: refusal counts fell only where the model COLLAPSED`.
A refusal direction should lower refusal at one sign and raise it at the other.
This one destroys the decoder at both signs, which means it is not a signed
refusal axis on this architecture.

## Mechanism

- Decreasing alpha raises KL and lowers refusal together (0.654/11, 0.986/9,
  1.388/7). The two move as one quantity.
- Attention-only coverage cuts KL 4.2x (1.388 -> 0.326) and removes **nothing**
  (12/12). The refusal effect and the damage both live in the same covered
  writes, so they cannot be separated by re-scoping the edit.
- Steering the direction collapses the model at both signs. There is no
  operating point, so the weight projection has nothing to inherit.

The honest summary: on LFM3-1.7B, this direction is a broad perturbation whose
apparent "refusal removal" is the perturbation sampled at a small enough dose to
keep the text coherent. The public "abliterated" LFM3 builds on the Hub are
therefore either doing something this pipeline does not measure, or they are
abliterated by exactly this mechanism at a dose nobody inspected.

## What would come next (declared, not run)

1. Re-derive the direction from **weight deltas**, not activations: contrastive
   weight steering (ft toward refusal, ft toward compliance, subtract) gives a
   direction that is defined on the same object the edit touches.
2. Harvest the refusal decision token with thinking **disabled**
   (`enable_thinking=false`) to see whether the thinking pathway is what makes
   the paired direction non-selective.
3. Per-layer selectivity instead of a global alpha: if refusal and damage share
   a locus, a per-layer projection with a measured damage budget per layer is
   the only way to find a non-empty window.

## Artifacts

- `campaigns/liquidai-lfm3-1.7b/directions-chat-paired-t96.pt` (205 KB)
- `campaigns/liquidai-lfm3-1.7b/directions-chat-paired-t320.pt` (205 KB)
- `campaigns/liquidai-lfm3-1.7b/steer-test-*.json` (full generations)
- `campaigns/liquidai-lfm3-1.7b/collect-*/bundle.json` + transcripts
- Checkpoints: `mpoa-a0.5-all24`, `mpoa-a1.0-all24`, `mpoa-a2.0-all24`,
  `mpoa-a2.0-attnonly` (3.4 GB each, re-derivable from the manifest in ~2 min)

Cleanup when no longer needed:

```bash
rm -rf ~/absolver/campaigns/liquidai-lfm3-1.7b/mpoa-*   # 13.6 GB of checkpoints
rm -rf ~/hf                                             # 3.4 GB model cache
```
