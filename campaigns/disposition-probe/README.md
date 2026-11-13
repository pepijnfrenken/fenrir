---
campaign: disposition-probe
model: LiquidAI/LFM2.5-1.2B-Instruct (+ our published abliterated build)
date: 2026-11-13
status: NEGATIVE (null — no greedy-level disposition shift detected)
method: activation-diff abliteration (upstream: lfm2.5-recovery / all32)
instrument: stated probability of a rise, forced integer, prefilled
---

# Rung 0 — is our own edit a different decision-maker?

Motivation: `arXiv:2607.17427` (*Abliteration Is Not a Scalpel*) shows that
removing refusal from a model shifts decision disposition — +12.2 pp (Gemma-4)
and +7.4 pp (Qwen3-30B) more optimistic up-bets on 21,600 decisions, in a task
that elicited **zero** refusals, so every delta is pure side effect. Our seven
gates (`gates.py`) cannot see this. This campaign asks whether the effect
reproduces on *our* published one-shot 1.2B edit, and whether it is attributable
to the removed direction or merely to perturbing the weights at that magnitude.

## Pre-registration (written before results, seed 20261113)

Three arms, identical item grid (120 constructed evidence bundles, tone-balanced
bullish/bearish/mixed, question order swapped per item, greedy, temp 0):

| arm | weights |
|---|---|
| `pristine` | `LiquidAI/LFM2.5-1.2B-Instruct`, official bf16 |
| `abliterated` | `PinoCookie/LFM2.5-1.2B-Instruct-Abliterated` (our 2026-09-02 push) |
| `noise` | pristine + Gaussian noise at matched per-tensor rel_l2 0.025 |

The noise arm is the attribution control: our replication measured the real
edit at rel_l2 0.022–0.030 per tensor, so a perturbation of that size with no
direction removed answers "is this the surgery or just the shake".

Primary endpoint: mean stated probability of a rise, paired per item.
Predictions:
- **P1** abliterated − pristine > 0 (more optimistic).
- **P2** noise − pristine ≈ 0 (shift is attributable to the direction).
- **P3** refusal rate 0 in every arm on this task (no refusals to remove, so any
  delta is disposition).
Falsifier: if P1 fails, `2607.17427` does not transfer to 1.2B one-shot
abliteration and the case for an 8th gate is much weaker.

## Instrument, declared

Not the paper's instrument and not a benchmark. Items are **constructed** from a
seeded template grid, not drawn. Forced integer probability with a prefilled
`PROBABILITY: ` so the answer is 3 tokens and parse cannot be confused with
prose. An earlier binary UP/DOWN version was discarded before use: the base
model answered UP 8/9 on bearish evidence (base-rate + position bias), which
saturates the instrument and cannot detect a shift. That is recorded here
because it is the same failure class as the keyword gate — an instrument that
cannot fail is not a gate.

**Headroom note (written after the pristine arm, before the other two).** The
pristine arm's stated probabilities occupy [28, 55]: bullish items sit at
exactly 55 (sd 0), mixed at 54.7 (sd 1.6), bearish at 42.5 (sd 7.7, n=37). So
the only headroom for an optimism shift is on the bearish subset, and a null
result on all 120 items would mean the instrument had no room, not that the
effect is absent. The bearish subset is pre-declared as the sensitive subset;
the primary endpoint stays the paired all-item mean, with the bearish subset
reported alongside.

## Results

Run 2026-11-13, 120 items × 3 arms, greedy, 0 unparsed / 0 non-integer / 0
refusals in any arm (P3 confirmed). `results/summary.txt`.

| arm | mean stated P(rise) | SE |
|---|---|---|
| pristine | 50.71 | 0.67 |
| abliterated | 50.70 | 0.69 |
| noise (rel_l2 0.025) | 50.52 | 0.74 |

Paired per item, abliterated − pristine:

| slice | n | delta | 95% CI (bootstrap B=2000) | sign test | items identical |
|---|---|---|---|---|---|
| all | 120 | −0.01 pp | [−0.72, +0.69] | +19/−18/0, 83 tied | 83/120 |
| bullish | 40 | +0.03 pp | [+0.00, +0.07] | +1/−0/39 | 39/40 |
| mixed | 40 | +0.00 pp | [−0.50, +0.55] | +9/−5/26 | 26/40 |
| bearish | 40 | −0.05 pp | [−2.00, +2.15] | +9/−13/18 | 18/40 |

noise − pristine, all items: −0.19 pp, CI [−1.07, +0.68], 103/120 identical.

**Verdict: P1 FAILS, P2 passes trivially, P3 passes.** Paired SD of the
difference is 4.0 pp, so the minimum detectable effect at 80% power is **1.0 pp**
on the all-item endpoint. The `2607.17427` effect is +7.4 to +12.2 pp on a
binary up-share. At that size, this instrument would have seen it and did not:
two thirds of the items (81/120) sit exactly on the model's 55% ceiling and are
byte-identical between arms, and the three arms are within 0.2 pp of each other
overall.

## What this does and does not show

It shows: our published 1.2B rank-1 edit, served at default greedy, states the
same probabilities as the pristine model on identical evidence, to within 1 pp.
The pessimism/optimism readout of `2607.17427` does **not** reproduce on it.

It does not show that abliteration is dispositionally inert, for three reasons
that are all mine, not the model's:

1. **Greedy only.** The paper sampled 5 per cell; sampling is where a
   distributional shift lives. Greedy reads one point of the distribution, and a
   shift in the tails would be invisible here. A sampled replication (T=0.7,
   5 samples/cell) costs ~5× this run and would settle it.
2. **Instrument headroom.** 81/120 items sit on a hard 55% ceiling. Only the
   bearish slice had room, and it carries the noisiest estimate (CI ±2 pp).
3. **Regime.** The paper's models are 26–30B MoE community checkpoints with
   all-layer edits; ours is a 1.2B hybrid with one rank-1 pass. If the effect is
   scale- or coverage-dependent, this run could not have found it.

The honest sentence is narrow: *at 1.2B, greedy, stated-probability instrument,
the off-target disposition effect does not appear.* Not "the effect is not
real".

## Next

- Rung 0b (optional, ~1.5 h CPU / ~10 min GPU): sampled replication, 5
  samples/cell, same grid, both arms. Only worth it if the disposition gate
  becomes a priority; the greedy null does not justify a gate on its own.
- Rung 1 (unchanged): the weight-arithmetic comparison in
  `docs/weight-space-editing-2026-11-13.md` §6 — that is the question this
  campaign was a prerequisite for, not a substitute for.


## Cleanup

```bash
rm -rf ~/absolver/campaigns/disposition-probe/models/noise-0.025   # 2.3 GB
rm -rf ~/absolver/.model-cache                                     # 2.3 GB
```
Results (`results/*.jsonl`) are small and stay.
