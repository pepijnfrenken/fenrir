# Weight-space editing vs fine-tuning — the 2026 map, and what it means for Fenrir

Written 2026-11-13. Scope: what "editing the weights instead of training them"
actually contains, what has been measured about it, and which of it Fenrir
already does under a different name. Every number below is attributed; anything
I could not open is marked UNVERIFIED and is not load-bearing.

## 0. The one-paragraph version

"Weight-space editing" is not one method, it is four, and Fenrir already lives
inside the first one. Abliteration *is* weight-space editing — a rank-1
projection applied to every matrix that writes to the residual stream. What is
new in the literature is a second family that derives the edit from *trained
weight deltas* instead of *measured activations* (contrastive weight steering,
task arithmetic), and a body of negative evidence about the price of both. The
evidence says three things worth acting on: (1) abliteration is not surgical —
it shifts decision disposition in ways our gates cannot see; (2) the
single-direction story is contested in activation space, but our own recovered
edit is empirically rank-1 in weight space, which is a different and stronger
claim; (3) fine-tuning, the thing weight editing is supposed to replace, is the
worse safety option, not the safer one.

## 1. Four families, disambiguated

| # | Family | Direction comes from | Applied to | Canonical refs |
|---|---|---|---|---|
| A | **Directional weight projection** (abliteration) | activation contrast (diff-in-means, paired, SVD, LEACE) | weights, at inference-free cost | Arditi et al. 2024, `arXiv:2406.11717`; Labonne 2024 |
| B | **Contrastive weight steering** (task arithmetic) | difference of two *fine-tuned* weight sets, `w_b = (θ⁺−θ_pre) − (θ⁻−θ_pre)` | weights, `θ + k·w_b` | Ilharco et al. ICLR 2023, `arXiv:2212.04089`; Fierro & Roger, `arXiv:2511.05408` (ICLR 2026) |
| C | **Locate-then-edit knowledge editing** | causal tracing → key/value slot in one MLP | a rank-one patch to one or a few MLP down-projections | ROME / MEMIT, Meng et al. 2022 |
| D | **Representation fine-tuning** (ReFT / LoReFT) | learned, trained intervention in an activation subspace | activations at inference, weights frozen | Wu et al. 2024 |

A and B are both weight arithmetic; they differ only in where the vector is
measured. A asks the model "which direction separates harmful from harmless
prompts", B asks "which direction separates a model trained to do X from a
model trained to do not-X". A needs activations, B needs two fine-tunes. That
distinction is the whole of the practical trade-off: B costs GPU-hours, A costs
a forward pass.

C is a different problem (facts, not dispositions) and its recent evidence is
bad — see §3. D is not a weight edit at all and should be read as the
parameter-efficient competitor to both fine-tuning and A.

## 2. What has actually been measured

**The single-direction claim (A's foundation).** Arditi et al. show refusal is
mediated by a direction that can be found by diff-in-means over harmful/harmless
prompts and removed either from activations or, permanently, by orthogonalizing
every matrix that writes to the residual stream. This is Fenrir's entire premise
and needs no defence here. `arXiv:2406.11717`.

**The multi-direction rebuttal, which is stronger than the claim it attacks.**
Four independent 2026 lines say the single direction is a compression, not the
mechanism:

- `arXiv:2502.17420` — refusal lives on "concept cones", not a line.
- `arXiv:2511.08379` (AAAI 2026) — SOM-extracted *multiple* refusal directions;
  ablating several beats the single-direction baseline and beats specialised
  jailbreak algorithms. They prove the SOM construction generalises diff-in-means.
- EMNLP 2026 main 2606, *There Is More to Refusal than a Single Direction* — the
  sharpest result for us: refusal categories are geometrically distinct, **yet
  steering along any refusal-related direction produces nearly identical
  refusal/over-refusal trade-offs**. The directions change *how* the model
  refuses, not *whether*. Linear intervention flattens the structure.
- `arXiv:2607.02396` (RFM-AGOP, ICML 2026 MI workshop) — fast multi-dimensional
  refusal subspace extraction in seconds, better ablation than alternatives.
  A search snippet of that paper's HTML claims rank-1 is insufficient at ≥8B and
  that ≥3 directions are needed to pass a 50% ASR threshold; **UNVERIFIED** — the
  HTML fetch returned a stub and the abstract does not contain it. Do not cite
  that number until someone reads the PDF.

Read together: at 0.3–3B a rank-1 edit flips refusal, which is exactly what our
campaigns show; the argument that rank-1 is insufficient is being made about
≥8B models, and it is being made about *activation* geometry. It does not yet
have a weight-space counterpart.

**Domain-specific abliteration at scale.** `arXiv:2607.02714` — 24 open models,
plus the first domain-scoped abliteration of a 1T-parameter MoE (Kimi K2),
removing cybersecurity refusal only. Two claims that matter: the refusal
subspace is distributed widely across *layers* in trillion-parameter MoEs, and
models fall into three susceptibility tiers predicted best by safety-training
type and architecture. `capability_map.py` answers a neighbouring question (where
capabilities sit relative to refusal, per layer) and does not slice refusal by
harm domain or produce a susceptibility tier, so neither claim is covered by
existing tooling.

**The price of A, measured where our gates are blind.** `arXiv:2607.17427`,
*Abliteration Is Not a Scalpel*: 21,600 decisions under uncertainty (weekly
up/down calls on 60 Warsaw equities over 18 weeks, 5 samples per cell, frozen
multi-agent pipeline, only the decision-layer model varying), BF16 official
checkpoints, one abliteration author (huihui), identical serving stack. The task
elicits **zero** refusals in the base arms — all 10,800 base decisions completed
without refusing — so every between-arm delta is pure side effect. Findings:
abliterated arms bet on the upside **+12.2 pp (Gemma-4-26B-A4B) and +7.4 pp
(Qwen3-30B-A3B-2507)** more often; justify themselves at greater length; use
fewer explicit uncertainty words with more concessive rhetoric; and expressed
confidence shifts in **opposite directions** across the two families. Capability
covariates rule out instruction-following degradation as the driver. One weight
surgery, two opposite disposition shifts.

That is the most important paper in this list for Fenrir, because our seven gates
cannot see any of it. A model can pass refusal ≤3%, coherence ≥90%, degeneracy
≤5%, finite logits, capability retention, PPL +15%, first-token KL 0.1 and still
be a measurably more optimistic decision-maker. We would publish it and call it
clean.

**The price of B, and why B is interesting.** `arXiv:2511.05408` (Fierro, MATS;
Roger, Anthropic), ICLR 2026. Contrastive weight steering with `w_b` as in §1,
`θ_steered = θ_pre + k·w_b`. Claimed results: steering often generalises
*further out of distribution* than activation steering before capability
degrades, because it edits all layers rather than one; for refusal, weight
steering is **as effective as including refusal data in fine-tuning**, and
outperforms activation steering; and the same `w_b` works as a *monitor* —
fine-tuning that induces emergent misalignment lands closer to an "evil" weight
direction than control directions. Note the asymmetry with A: B's vector is
built by subtraction of two deltas, so topic/style/length differences cancel,
leaving the behaviour.

**The alternative is worse, which is the actual argument for editing.**
`arXiv:2310.03693` (Qi et al., ICLR 2024): safety alignment is compromised by
fine-tuning on as few as **10 adversarially designed examples**; the authors'
own policy brief prices this under $0.20 of API calls, and notes that even
benign fine-tuning degrades safety. Anyone arguing "just fine-tune it back" is
arguing for the cheaper attack.

**And the sibling family's evidence is bad.** `arXiv:2510.00625`, *Is Model
Editing Built on Sand?*: the fundamental objective of editing — steer the output
toward a target with minimal modification — rewards exploiting hidden shortcuts
rather than real semantics; state-of-the-art editors **collapse under the
simplest negation queries**, because the benchmark suites lack negative examples.
This does not transfer directly to A or B (we do not claim to install knowledge),
but it is a warning about the evaluation pattern: an edit that passes its own
probe distribution has proven nothing about the distribution you did not probe.

## 3. What Fenrir already has, precisely

Two repo facts, both measured, both already in the campaign KB:

1. **Our method set is A.** `excise.EXCISE_REALIZED_METHODS =
   {mpoa, projection, advanced, direct_ablation, stacked_ablation}`. `config.py`
   also declares `steering`, `lora`, `bias_vectors`, `projected` — `sweep.py`
   drops any method outside the realized set with a warning and falls back to
   the realized list. So those four are never evaluated: the *default* sweep box
   advertises six methods and searches two of them. Nothing in the pipeline
   does B or C.

2. **Our published edit is rank-1 in weight space.** From the
   `lfm2.5-recovery` campaign: for every out-projection, `ΔW = W_abl − W_base`
   has 99.4% of its energy in σ₁; the top left-singular vector `u₁` is shared
   across a layer's out-projections at |cos| 0.99+; re-applying `σ₁u₁v₁ᵀ`
   reproduces the published model at rel_l2 0.001–0.002. Result at default
   greedy: refusal 0/5 with verbatim compliance, PPL +5.8%, MMLU retention 1.0.

Point 2 is the finding this literature does not have. The multi-direction
argument is made against *activation* geometry; our recovered artifact shows that
for this architecture and this edit, the *weight*-space object is rank-1 and a
rank-1 re-application is behaviourally sufficient. That is a narrow, honest,
falsifiable claim with a fingerprint attached — exactly the kind of thing worth
writing up rather than the fifth opinion about whether refusal is a line.

## 4. The gaps that this research exposes

- **G1 — no disposition instrument.** Seven gates, none of which can see
  §2's +7.4/+12.2 pp. This is the same failure class as the keyword gate: an
  instrument that cannot see the behaviour sitting next to it. Cheapest fix is
  a probe, not a paper: a forced-choice-under-uncertainty bank + uncertainty
  vocabulary rate + self-justification length, run on pristine and edited arms.
  It does not need a market, only identical evidence and a binary call.
- **G2 — no trained-delta path.** B needs two narrow fine-tunes. The only
  AdamW/backward in the repo is `campaigns/lfm2.5-2.6b/lora-repair-modal.py`.
  There is no target-SFT trainer, so B cannot be tested without writing one.
- **G3 — rank-1 is unexamined at ≥8B.** We have never measured whether σ₁
  energy share survives scale, or whether our stacking recipe still passes
  gates at 8B+. The literature says the activation geometry changes there; we
  have no weight-space measurement either way.
- **G4 — declared-but-unrealizable methods** (§3.1). Either implement them in
  `excise.py` or delete them from `config.py`. A declared method that raises at
  finalization is a trap for the next campaign.

## 5. Rung 0 — the experiment I would run first

Minimal scope, no training, no GPU if the model fits: **measure G1 on our own
published artifact.** Three arms: pristine LFM2.5-1.2B-Instruct, our published
abliterated build, and — as the control that makes it a measurement rather than
a vibe — the same model with a fixed benign-noise weight perturbation of matched
rel_l2. If the abliterated arm moves optimism/uncertainty relative to pristine
but the noise arm does not, the disposition shift is attributable to the edit,
not to perturbation as such. Built and running in
`campaigns/disposition-probe/`; the pre-registration, the discarded first
version of the instrument, and the headroom limit are all recorded there.

Pre-registered predictions, so the run can lose:
- P1: abliterated arm is more optimistic than pristine on identical evidence.
- P2: the noise control does not move the optimism rate.
- P3: refusal rate is 0 in the edited arm — i.e. the task elicits no refusals,
  so any delta is pure side effect (the `2607.17427` design).
Falsifier: if P1 fails on our edit, `2607.17427` does not transfer to 1.2B
one-shot abliteration and the disposition gate is lower priority than I am
arguing.

**Outcome (2026-11-13): P1 failed.** abliterated − pristine = −0.01 pp, 95% CI
[−0.72, +0.69], paired n=120, MDE(80%) 1.0 pp; 83/120 items answered
identically; zero refusals in every arm. The noise control moved nothing either
(−0.19 pp, CI [−1.07, +0.68]). So on a 1.2B rank-1 edit at default greedy the
`2607.17427` disposition effect does not appear. Three limits are mine, not the
model's — greedy-only (the paper sampled 5/cell), a 55% ceiling that left
headroom on only the bearish third of the grid, and a much smaller model with
one pass rather than 26–30B MoE all-layer edits. Full write-up:
`campaigns/disposition-probe/README.md`. The gate question is open, not settled;
this does not justify building the gate and it does not refute the paper.

## 6. Rung 1 — the actual weight-arithmetic test, if rung 0 lands

Two narrow LoRA fine-tunes on the same 200 paired prompts used for PROBE (one
adapting toward compliance, one toward refusal), `w_b = τ⁺ − τ⁻`, then
`θ + k·w_b` at k ∈ {0.25, 0.5, 1.0, 2.0} with k chosen on the tune split.
Compare against `stacked_ablation` at matched held-out refusal. Name the price
before the run: capability retention, PPL increase, first-token KL, and the §5
disposition numbers on both arms. The claim to beat is not "refusal goes to
zero" — both will do that. It is "which one buys refusal at lower capability
cost and smaller disposition shift".

Cost: 2 LoRA runs (~1 GPU-hour at 1.2B) plus one gate bundle. Bounded,
reversible, offline.

## Provenance

| Claim | Source | Read or snippet |
|---|---|---|
| rank-1 weight edit behaviourally sufficient (LFM2.5) | `campaigns/lfm2.5-recovery/README.md` | repo measurement |
| realized method set excludes steering/lora/bias_vectors/projected | `excise.py:28` | repo code |
| +12.2 / +7.4 pp optimism, 21,600 decisions, 0 base refusals | `arXiv:2607.17427` abstract + key findings | full abstract |
| weight steering ≥ activation steering OOD; refusal ≈ FT-with-refusal-data | `arXiv:2511.05408` abstract | full abstract |
| steering along any refusal direction ≈ same refusal/over-refusal trade-off | EMNLP 2026 main 2606 abstract | full abstract |
| multi-direction beats single-direction ablation | `arXiv:2511.08379` abstract | full abstract |
| 10 examples break safety, <$0.20 | `arXiv:2310.03693` + Stanford HAI policy brief | abstract + brief |
| SOTA knowledge editing collapses under negation | `arXiv:2510.00625` abstract | full abstract |
| rank-1 insufficient ≥8B, ≥3 directions for 50% ASR | `arXiv:2607.02396` HTML snippet | **UNVERIFIED — not load-bearing** |
