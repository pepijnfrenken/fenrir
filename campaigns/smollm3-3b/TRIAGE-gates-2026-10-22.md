# Lane B triage: SmolLM3-3B first-look gate bundle (run `462109`)

Date: 2026-10-22. Read from the repo code at `13a3d62` plus the run's gate dict. No recompute
was possible tonight (cluster access blocked, see the blocker log at the end).

## Observed bundle

```
refusal                True   0.0
degeneracy             True   0.0
finite_logits          True   1.0
coherence              False  0.0
capability             False  0.0
perplexity_increase    False  None
first_token_kl         False  None
baseline_sanity        False  0.0
eval_pass              False
```

MaxRSS 37.2 GB. Output `abliterated_SmolLM3-3B`. Chain completed: PROBE (`refusal_layers=36`,
`affirm_layers=36`) then DISTILL (`diff_means`, 1 direction, 36 layers scored) then EXCISE
(`projection`, alpha 1.00, `layers=[0..35]`, `weights=['o_proj']`).

## Finding 1: four of the five red gates are a scaffolding cascade, not model damage

`verify.py` gates the ENTIRE pristine-baseline block on benchmark scores:

```python
if (getattr(cfg, "verify_pristine_baseline", True)
    and pristine
    and requested
    and benchmark_scores):        # verify.py:1012-1016
```

Inside that block sit: `pristine_logprobs` (PPL gate input), `pristine_logprobs_first`
(first-token KL gate input), `pristine_scores` (capability retention), `pristine_refusal`
(baseline sanity), and the `logger.info("PRISTINE refusal rate: ...")` line (verify.py:1134).

`benchmark_scores` is built from `mmlu` (only when `verify_sample_size > 0`) plus
`requested = [name for name in cfg.model_card_targets if name in _RUNNER]` (verify.py:967-975).
The registered runners are mmlu, arc_easy, bbh, hellaswag, truthfulqa, math, mmlu_pro, gpqa,
aime. For this locally written yaml neither path produced anything, so `benchmark_scores` came
out empty. Downstream, exactly as designed:

- `gate_capability` returns `value 0.0, passed False, detail "no benchmark scores available"`
  (gates.py:305-310). The recorded `capability False 0.0` is that branch, NOT a retention of 0.
- `gate_perplexity_increase(pristine_logprobs=None)` returns `value None, passed False,
  "no pristine baseline; gate skipped (skipped != passed)"` (gates.py:240-245).
- `gate_first_token_kl` does the same on an empty `pristine_logprobs_first` (gates.py:345).
- `baseline_sanity` needs `pristine_refusal`, which is None because the block never ran
  (verify.py:1204-1210, "no pristine axis recorded -> red").

So one missing benchmark configuration turns four gates red. That is a configuration defect in
the yaml, not evidence that the edit broke the model, and it repeats the class of error the
toolkit already documents (a skipped gate must never report green, and here it reports red for
the wrong reason).

**Fix (no judge key needed):** set `verify_benchmarks: ["mmlu"]` (or `verify_sample_size: 32`)
in `models/smollm3-3b.yaml` and re-run VERIFY. All four gates then have their inputs. None of
them requires the missing LLM judge.

## Finding 2: the edit itself deviates from the recipe, so `coherence 0.0` may be real

The run excised **every layer 0 to 35** on `o_proj` alone at alpha 1.00. The documented guard /
chat recipe is a BAND window plus both write matrices (`o_proj` and `down_proj`), because:

- single layers are insufficient (cross-layer redundancy), and
- full-band edits destroy selectivity and coherence (the "guard starts flagging everything"
  failure mode, and in a chat model the analogous collapse).

A full-stack `o_proj`-only ablation at alpha 1.0 is the maximal-removal shape, so a coherence
read of 0.0 is plausible as genuine damage rather than another scaffold artifact. It cannot be
distinguished from a scaffold problem from the current bundle alone, because `coherence 0.0`
was recorded with `perplexity_increase = None`: there is no second, threshold-free measure of
text quality in the bundle to cross-check against.

**Order of business, not to be swapped:** re-run with the yaml fix first (cheap, resolves the
cascade). If `coherence` is still 0.0 with PPL and KL computed, the edit is the cause, and the
next step is a band ladder (mid-stack and late-band windows, `o_proj` + `down_proj`, alpha
0.75 / 1.0 / 1.25) rather than a full-stack sweep. The same ladder shape as the g06 guard lane,
run through the new driver where the model family allows.

## Finding 3: compliance claims remain gated on the judge

No `FREEINFERENCE_API_KEY` / `JUDGE_API_KEY` was found anywhere on the cluster after
`bw_load_env`, so behavior verdicts fall back to keyword-only labels (`evasive_nojudge`). Per
the toolkit rule, those are weak evidence only. `refusal True 0.0` is a refusal count, which is
judge-independent, so it can be quoted; anything about "compliance" cannot.

## Blocker log

2026-10-22 22:13 UTC: the TU/e eduVPN session expired (profile expiry 2026-10-22T22:13:23Z) and
the `tue` WireGuard interface stopped passing traffic. Re-issuing needs the portal plus
`wg-quick up` as root. Until then nothing on this page can be re-run.
