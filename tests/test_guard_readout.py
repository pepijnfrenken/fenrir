"""Regression tests for the guard-mode readout + gates (guard lane, 2026-09-15).

Contracts defended:
- verdict parsing across families (qwen3guard CALIBRATED; others BLIND-marked)
- decision-step location for the verdict margin (the step right after "Safety:")
- verdict-token resolution incl. config-supplied ids
- two-readout judgement: label + margin + divergence bookkeeping
- run_gates guard branch: chat-behavior gates skipped (never green), guard
  gates + guard baseline sanity wired, eval_pass fails closed on unmeasurable
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig  # noqa: E402
from gates import run_gates  # noqa: E402
from guard_readout import (  # noqa: E402
    decision_step_index,
    parse_verdict,
    resolve_verdict_tokens,
    verdict_read,
)

# --------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------- #

QWEN_SAMPLES = [
    ("Safety: Unsafe\nCategories: Violent\nRefusal: No", "Unsafe", "flag", 1.0, True),
    ("Safety: Safe\nCategories: None\nRefusal: Yes", "Safe", "safe", 0.0, False),
    ("Safety: Controversial\nCategories: Politically Sensitive Topics", "Controversial",
     "controversial", 0.5, False),
]


@pytest.mark.parametrize("text,verdict,band,score,flagged", QWEN_SAMPLES)
def test_parse_qwen3guard(text, verdict, band, score, flagged):
    j = parse_verdict(text, "qwen3guard")
    assert (j.verdict, j.band, j.score, j.flagged) == (verdict, band, score, flagged)
    assert j.calibrated is True


def test_parse_qwen3guard_refusal_field_and_categories():
    j = parse_verdict("Safety: Unsafe\nCategories: Non-violent Illegal Acts, Unethical Acts\nRefusal: No")
    assert j.categories == ("Non-violent Illegal Acts", "Unethical Acts")
    assert j.refusal == "No"


def test_parse_other_families():
    j = parse_verdict("unsafe\nS1,S2", "llama_guard")
    assert j.flagged and j.band == "flag" and "S1" in j.categories and j.calibrated is False
    j = parse_verdict("safe", "llama_guard")
    assert not j.flagged and j.band == "safe"
    j = parse_verdict("Yes", "shieldgemma")
    assert j.flagged and j.band == "flag"
    j = parse_verdict("No", "granite_guardian")
    assert not j.flagged and j.band == "safe"


def test_parse_unparsed_and_unknown_family():
    j = parse_verdict("I cannot decide.", "qwen3guard")
    assert j.band == "unparsed" and j.score is None and j.flagged is False
    # unknown family falls back to the default, never crashes
    j = parse_verdict("Safety: Unsafe", "no_such_family")
    assert j.family == "qwen3guard" and j.flagged


# --------------------------------------------------------------------- #
# Decision-step location + token resolution (pure)
# --------------------------------------------------------------------- #

def test_decision_step_index():
    assert decision_step_index(["Safety", ":", " Unsafe"], "Safety:") == 2
    assert decision_step_index(["Safety", ":", " Unsafe"], None) == 0
    assert decision_step_index(["no marker", "here"], "Safety:") is None
    assert decision_step_index([], "Safety:") is None
    # marker completing across the raw string with trailing space
    assert decision_step_index(["Safety:", " Unsafe"], "Safety:") == 1


class _EncTok:
    """Tokenizer stub: single-token variants map to ids; other strings are multi-token."""
    _map = {" Safe": 10, "Safe": 10, " Unsafe": 11, "Unsafe": 11,
            " Controversial": 12, "Controversial": 12}

    def encode(self, s, add_special_tokens=False):
        return [self._map[s]] if s in self._map else [20, 21, 22]


def test_resolve_verdict_tokens_merges_config_ids():
    toks = resolve_verdict_tokens(_EncTok(), {"unsafe": [11, 99], "safe": []})
    assert toks["unsafe"][0] == 11 and 99 in toks["unsafe"]
    assert len(toks["unsafe"]) == len(set(toks["unsafe"]))  # deduped
    assert 10 in toks["safe"]
    assert toks["controversial"] == [12]


# --------------------------------------------------------------------- #
# Model-side read (stub model + tokenizer)
# --------------------------------------------------------------------- #

PIECES = {1: "Safety", 2: ":", 10: " Safe", 11: " Unsafe"}


class _GOut:
    def __init__(self, seq, scores):
        self.sequences = seq
        self.scores = tuple(scores)


class _GuardStub(torch.nn.Module):
    """Returns a fixed verdict-token stream; logits come from the caller."""

    def __init__(self, new_ids, scores):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self._new_ids = new_ids
        self._scores = scores

    def generate(self, **kw):
        base = kw["input_ids"][0].tolist()
        return _GOut(torch.tensor([base + self._new_ids]), self._scores)


class _Batch(dict):
    def to(self, dev):
        return self


class _GuardTok:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<prompt>"

    def __call__(self, text, **kw):
        return _Batch(input_ids=torch.zeros(1, 4, dtype=torch.long))

    def encode(self, s, add_special_tokens=False):
        m = {" Safe": 10, "Safe": 10, " Unsafe": 11, "Unsafe": 11,
             " Controversial": 12, "Controversial": 12}
        return [m[s]] if s in m else [20, 21]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(PIECES.get(i, "") for i in ids)


def _scores_with(step_logits):
    """One score tensor per step; vocab 32."""
    out = []
    for vals in step_logits:
        t = torch.full((1, 32), -5.0)
        for tid, v in vals.items():
            t[0, tid] = v
        out.append(t)
    return out


def test_verdict_read_flag_with_margin():
    scores = _scores_with([{}, {}, {11: 6.0, 10: 1.0}])  # step 2: Unsafe wins
    model = _GuardStub([1, 2, 11], scores)
    out = verdict_read(model, _GuardTok(), [{"role": "user", "content": "x"}])
    assert out["verdict"] == "Unsafe" and out["band"] == "flag"
    assert out["decision_step"] == 2
    assert out["margin"] == pytest.approx(5.0)
    assert out["divergent"] is False
    assert out["top5"] and " Unsafe" in out["top5"]


def test_verdict_read_divergence_is_recorded():
    # TEXT says Safe, but the logits say Unsafe -> divergence must be flagged
    scores = _scores_with([{}, {}, {11: 6.0, 10: 1.0}])
    model = _GuardStub([1, 2, 10], scores)
    out = verdict_read(model, _GuardTok(), [{"role": "user", "content": "x"}])
    assert out["verdict"] == "Safe" and out["band"] == "safe"
    assert out["margin"] > 0
    assert out["divergent"] is True


def test_verdict_read_marker_missing_reports_reason():
    scores = _scores_with([{}, {}])
    model = _GuardStub([7, 8], scores)  # pieces 7/8 are not in PIECES -> empty decode
    out = verdict_read(model, _GuardTok(), [{"role": "user", "content": "x"}])
    assert out["margin"] is None and "marker" in (out["reason"] or "")
    assert out["divergent"] is None


# --------------------------------------------------------------------- #
# Guard-mode run_gates branch
# --------------------------------------------------------------------- #

def _guard_pairs():
    return [
        {"pair_id": "h1", "prompt": "p1", "response": "r1", "expected": "Unsafe"},
        {"pair_id": "h2", "prompt": "p2", "response": "r2", "expected": "Unsafe"},
        {"pair_id": "b1", "prompt": "p3", "response": "r3", "expected": "Safe"},
        {"pair_id": "b2", "prompt": "p4", "response": "r4", "expected": "Safe"},
    ]


def test_run_gates_guard_mode_wiring_and_fail_closed():
    cfg = ModelConfig(model_id="stub")
    cfg.guard_mode = True
    # guard reads everything Safe -> flag rate 0; on a PRISTINE run that means
    # no measurable axis -> baseline sanity red, eval_pass False.
    scores = _scores_with([{}, {}, {10: 6.0, 11: 1.0}])
    model = _GuardStub([1, 2, 10], scores)
    tl: list[dict] = []
    report = run_gates(model, _GuardTok(), cfg, prompts=["p"], benchmark_scores={},
                       is_pristine=True, guard_pairs=_guard_pairs(), transcript=tl)
    assert report["_guard_mode"] is True
    assert "refusal" not in report  # chat gates were skipped, not run
    assert set(report["_skipped"]) >= {"refusal", "coherence", "degeneracy"}
    assert report["flag_rate"]["value"] == 0.0
    assert report["flag_rate"]["passed"] is True   # a 0 flag rate is the SUCCESS state for an ablation
    assert report["pass_rate_benign"]["value"] == 1.0
    assert report["baseline_sanity"]["passed"] is False  # ...but unmeasurable on a pristine run
    assert report["eval_pass"] is False
    assert len(tl) == 4  # two reads per pair group (harmful + benign), 2 pairs each
