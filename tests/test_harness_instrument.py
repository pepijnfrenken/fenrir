"""Tests for the harness instrument fixes (ABSOLVER-1).

Covers the pieces that are pure logic (no model load):
- directions filename keyed on flavor + dir_method (a paired harvest used to
  overwrite the diff_means harvest: campaigns/minicpm5-2b bugs table)
- local model paths never hit the hub in `_copy_trust_remote_code` (401 warning
  per `abl` run on a local symlinked model_id)
- the refusal window helper (>=256 tokens; 64 ends inside the thinking preamble)
- the steer-test PRE-EDIT causality verdict (`--require-effect`)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from harness.abl import (
    _copy_trust_remote_code,
    _directions_filename,
    _is_local_model_path,
    _steer_causality,
)
from verify import _refusal_max_new_tokens


# --------------------------------------------------------------------------- #
# directions filename
# --------------------------------------------------------------------------- #

def test_directions_filename_keys_on_flavor_and_method():
    """paired must not overwrite diff_means in the same campaign dir."""
    paired = _directions_filename("chat", "paired")
    diff = _directions_filename("chat", "diff_means")
    assert paired != diff
    # matches the desktop's hand-renamed workaround, standardized in code
    assert paired == "directions-chat-paired.pt"
    assert diff == "directions-chat-diff_means.pt"
    assert _directions_filename("raw", "paired") != paired


# --------------------------------------------------------------------------- #
# local model paths must not be sent to the hub
# --------------------------------------------------------------------------- #

def test_is_local_model_path(tmp_path, monkeypatch):
    local = tmp_path / "MiniCPM5-2B"
    local.mkdir()
    assert _is_local_model_path(str(local)) is True
    link = tmp_path / "MiniCPM5-2B-link"
    link.symlink_to(local)
    assert _is_local_model_path(str(link)) is True
    assert _is_local_model_path("openbmb/MiniCPM5-2B") is False
    assert _is_local_model_path("") is False


def test_copy_trust_remote_code_local_path_does_not_touch_the_hub(tmp_path, monkeypatch):
    """The 401/no-download contract for local model dirs."""
    model_dir = tmp_path / "MiniCPM5-2B"
    model_dir.mkdir()
    (model_dir / "modeling_minicpm.py").write_text("# custom code\n", encoding="utf-8")
    out_dir = tmp_path / "ablated"
    out_dir.mkdir()

    def _boom(*a, **kw):  # a hub call here is the bug
        raise AssertionError("snapshot_download called for a local model path")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=_boom))
    cfg = types.SimpleNamespace(model_id=str(model_dir))
    _copy_trust_remote_code(cfg, out_dir)
    assert (out_dir / "modeling_minicpm.py").read_text(encoding="utf-8") == "# custom code\n"


def test_copy_trust_remote_code_local_dir_without_py_files(tmp_path, monkeypatch):
    model_dir = tmp_path / "plain"
    model_dir.mkdir()
    out_dir = tmp_path / "ablated"
    out_dir.mkdir()

    def _boom(*a, **kw):
        raise AssertionError("snapshot_download called for a local model path")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=_boom))
    cfg = types.SimpleNamespace(model_id=str(model_dir))
    _copy_trust_remote_code(cfg, out_dir)
    assert list(out_dir.iterdir()) == []


def test_copy_trust_remote_code_hub_id_still_downloads(tmp_path, monkeypatch):
    src = tmp_path / "snapshot"
    src.mkdir()
    (src / "configuration_x.py").write_text("# hub code\n", encoding="utf-8")
    out_dir = tmp_path / "ablated"
    out_dir.mkdir()
    seen: list[str] = []

    def _snapshot(model_id, *a, **kw):
        seen.append(model_id)
        return str(src)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=_snapshot))
    cfg = types.SimpleNamespace(model_id="org/custom-model")
    _copy_trust_remote_code(cfg, out_dir)
    assert seen == ["org/custom-model"]
    assert (out_dir / "configuration_x.py").exists()


# --------------------------------------------------------------------------- #
# refusal window
# --------------------------------------------------------------------------- #

def test_refusal_window_defaults_wide():
    """64 tokens ends inside the thinking preamble — never the default."""
    assert _refusal_max_new_tokens(types.SimpleNamespace()) >= 256
    assert _refusal_max_new_tokens(types.SimpleNamespace(gate_refusal_max_new_tokens=512)) == 512
    assert _refusal_max_new_tokens(types.SimpleNamespace(gate_refusal_max_new_tokens=0)) == 0


# --------------------------------------------------------------------------- #
# steer-test causality verdict
# --------------------------------------------------------------------------- #

def _row(alpha: int, refusals: int, n: int = 5) -> dict:
    return {"alpha": float(alpha), "refusal_count": refusals, "n_prompts": n}


def test_steer_causality_flags_a_direction_that_never_moves_refusal():
    """minicpm5-2b's measured shape: every alpha still refuses."""
    v = _steer_causality([_row(0, 5), _row(-20, 5), _row(-5, 5), _row(5, 5), _row(20, 5)])
    assert v["causal"] is False
    assert v["effect_alphas"] == []
    assert v["baseline_refusals"] == 5
    assert "NOT CAUSAL" in v["verdict"]


def test_steer_causality_detects_a_real_effect():
    v = _steer_causality([_row(0, 5), _row(1.0, 5), _row(2.0, 3), _row(3.0, 0)])
    assert v["causal"] is True
    assert v["effect_alphas"] == [2.0, 3.0]


def test_steer_causality_requires_a_baseline_row():
    """No rows -> unknown, and never a silent pass."""
    v = _steer_causality([])
    assert v["causal"] is False
    assert "unknown" in v["verdict"]


def test_steer_causality_ignores_worse_alphas():
    """A direction that INCREASES refusal is not causal."""
    v = _steer_causality([_row(0, 2), _row(1.0, 5), _row(2.0, 4)])
    assert v["causal"] is False
