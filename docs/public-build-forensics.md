# Public-build forensics

**When to use this.** Your own recipe has failed on a model more than twice, and
public "abliterated" builds of the same model exist. Do this before running
another alpha ladder. A ladder explores your parameterisation. Forensics finds
out whether your parameterisation is the right one.

This document is the procedure, the numbers it produces, and the four failures it
caught in one night on `LiquidAI/LFM3-1.7B`.

---

## 1. Enumerate what actually exists

Query the Hub by search, not by memory:

```
GET https://huggingface.co/api/models?search=<model>&limit=100
GET https://huggingface.co/api/models/<repo>            # siblings, tags, cardData
GET https://huggingface.co/api/models/<repo>?blobs=true # file sizes
GET https://huggingface.co/<repo>/raw/main/README.md
```

Two filters decide whether a build is usable evidence:

- **Weights, not just GGUF.** A diff needs `*.safetensors`. A GGUF-only repo
  cannot be compared tensor by tensor without dequantising.
- **Disclosed parameters.** The card may state the search space, the layer
  window, the KL budget, or the tool and version. Undisclosed download counts are
  popularity, not evidence.

Tags often disclose the method outright. `arbitrary-rank-ablation`, `heretic`,
`kld-0.0374`, `refusal-removed` appeared in tags before the prose did.

## 2. Read the tool's source, not the card's summary

When a card names a tool (`heretic 1.4.0`, and 2.1.0 shipped on PyPI), download
the wheel and read the modifier. `pip download heretic-llm --no-deps` then
`unzip` is enough to establish:

- the direction harvest: which token position, which prompt sets, whether the
  direction is orthogonalised against the good-mean direction;
- the edit: `dW = -lambda v (v^T W)`, so rank-1 by construction, with row
  magnitudes restored (the same family as MPOA);
- the parameterisation: `direction_index` is fractional and builds one global
  direction by lerp between neighbouring layer directions; `lambda` per component
  follows a tent (`max_weight` at `max_weight_position`, decaying to `min_weight`
  at `min_weight_distance`).

A card tells you what the author wants you to know. The source tells you what the
edit is.

## 3. Diff the weights

`campaigns/liquidai-lfm3-1.7b/desktop/diff_public.py --base <base> --other <build>`

Per changed tensor it reports `rel_change`, `rank1_frac` (s1 / sum of singular
values), effective rank, the top five singular values, and the maximum relative
row-norm drift. Four readings follow from that, and they are different questions:

| measurement | what it answers |
|---|---|
| `rel_change` | how big the edit is — compare with your own edits before blaming anything else |
| `rank1_frac`, effective rank | is the edit a ray or a subspace? |
| top left singular vector across layers | do the layers carry *different* edits, or one axis repeated? |
| row-norm drift | is row-magnitude preservation actually held? |

### What that produced on LFM3-1.7B

`Austriani/LFM3-1.7B-ARA-Abliterated` against `LiquidAI/LFM3-1.7B`:

- **11 tensors changed**: `o_proj` and `w2` only, layers 12, 14, 15, 16, 17, 18,
  19, 22. No `conv_out`. Not all 24 layers.
- `rel_change` **4.04 % – 14.09 %** — the same order as this campaign's α 2.0
  edit (4.8 % on `w2`), so dose was never the difference.
- `rank1_frac` **0.205 – 0.644**, effective rank **2.3 – 11.3** — a subspace, not
  a ray.
- row-norm drift up to **18.0 %** (L19 `w2`) — magnitude preservation is *not*
  held exactly, so it cannot be the whole mechanism either.
- **cross-layer cosine of the top left singular vector: mean −0.04, |max| 0.19.**
  Each layer carries its own direction.

The campaign's own 24 per-layer directions measured **0.49 mean cosine across
layers (0.81 between neighbours)**. So "all 24 layers" was one global axis applied
24 times: the same removal repeated, and therefore uniform damage. That is a
structural difference between two edits of the same size, and no alpha sweep can
find it.

## 4. Gate the public build with your own instruments

Do not accept the published refusal number. Published "6/100 refusals" may come
from a keyword scorer, and a keyword scorer cannot read a thinking model.

Two mechanical requirements, both learned the hard way:

1. **Pass a foreign build as `--model-dir`, not as the config's `model_id`.**
   `collect` treats `model_id` as pristine, and the baseline sanity gate then
   aborts with `BASELINE SANITY GATE RED — no refusal signal`. That guard is
   correct for a pristine model and wrong for the object under test.
2. **Set the window so the thinking block closes.** At 1024 tokens every one of
   the 12 responses hit the cap mid-reasoning (`<｜end▁of▁thinking｜>` absent,
   4292–4841 characters). "No refusal" was measured; "answers or slop" was not.
   At 4096 tokens the responses finished (220–11190 characters).

### The measurement that came out

`MihaiPopa-1/LFM3-1.7B-Heretic-Max`, 12-prompt held-out split, 4096 tokens:

| gate | result |
|---|---|
| refusal | **0/12** style-aware, 0/12 keyword |
| coherence | 12/12 |
| degeneracy | 0.000 |
| capability | MMLU 0.500 vs pristine 0.400 (retention 1.25) |
| first-token KL (harmful split, legacy protocol) | 11.73 nats |
| semantic judge, 12 responses | **6 ANSWER, 5 GARBLED, 1 disagree**, agreement 11/12 |

So the build works and is not clean. It removes refusal everywhere, keeps the
model coherent and its benchmark score, and still produces unusable output on
**5 of 12** prompts. Published refusal counts do not capture that half.

## 5. The instrument defect this exposed

`first_token_kl` and `perplexity_increase` were measured on the **held-out harmful
prompts** — the prompts whose distribution is the thing being changed. A build
that removes refusal therefore fails the KL gate by construction, and a build that
still refuses passes it. The public build scored 11.73 nats while being the best
object this campaign has measured; the campaign's own failing edits scored ~0.3
and looked cleaner.

**Interference is measured where behaviour must not move.** Refusal is measured
where it must. Never the same set.

Enforcement is now structural, not advisory:

- `config.n_interference_prompts` (default 20) selects a harmless pool;
  `cmd_collect` computes pristine baselines on the held-out split **and** that
  pool, and passes it to `run_gates(interference_prompts=...)`.
- `gates.interference_prompt_set()` labels every interference number with the set
  that produced it (`prompt_set` field in the bundle).
- `gates.enforce_interference_provenance()` **forces `eval_pass: false`** when
  either interference gate was measured on the legacy harmful set, whatever the
  individual gates say, and records `_eval_pass_blocked_by`. Opting out requires
  `gate_require_harmless_interference: false` in the config, and the bundle then
  records `_legacy_interference_accepted`.
- Regression test:
  `test_a_green_run_cannot_rest_on_legacy_interference`.

## 6. Reproducing the published recipe on your own weights

`campaigns/liquidai-lfm3-1.7b/desktop/reproduce_heretic.py` reimplements the
disclosed harvest and applies the disclosed profile, so the weights are yours and
the verdict is yours:

- residuals at the **first generated token**, 400 harmful vs 400 harmless, not the
  mean over all response tokens;
- direction = `mean(bad) - mean(good)`, normalised, **orthogonalised against the
  normalised mean(good) direction**;
- one global direction by lerp at a fractional layer index (11.53 → 0.53 of the
  way from L11 to L12);
- `lambda` per component from the tent, applied to `o_proj` + `w2`, layers 11–23,
  row magnitudes restored.

Declared falsifiers live in the campaign README (rung 8): if cos(their direction,
our paired direction) < 0.5 the two harvests are different objects; if our meter
reads KL ≥ 0.5 at the disclosed lambda, the published KL is not reproducible here.

## 7. Rules that came out of this

1. A failing sweep is not evidence about the model. It is evidence about your
   parameterisation. Forensics answers the second question; another ladder does
   not.
2. Compare doses before you compare methods. `rel_change` first, always.
3. Rank-1 versus subspace is measurable in one SVD, and it decides whether a
   layer profile means anything.
4. Cross-layer cosine decides whether "all layers" is 24 edits or one edit
   repeated 24 times.
5. A refusal count without a semantic label is a claim about the absence of a
   sentence, not about the presence of an answer.
6. Interference on harmless, refusal on harmful. A gate that moves with the effect
   is not measuring damage.
7. Published numbers are measured with unknown instruments. Re-measure before you
   use one as a target.
