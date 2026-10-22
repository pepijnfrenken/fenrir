"""Offline self-test for guard_excise_ladder.py (no model, no GPU, no cluster).

Verifies the three things that can be checked without loading a guard:
  1. messages_for reproduces the guard's two native reads;
  2. aggregate + gate_flags compute the class split, AUC and refit recovery;
  3. apply_edit's alpha is the FRACTION of the d-projection removed from the
     weight (alpha=1.0 nulls d^T W d), which is the unit that steering does NOT
     use. This is the check that catches copying a steer alpha into excise.
Run: .venv/bin/python campaigns/guard-lane/test_excise_ladder_offline.py
"""
import importlib.util
import sys
import types

import numpy as np
import torch

PATH = "/home/pino/absolver/campaigns/guard-lane/guard_excise_ladder.py"
spec = importlib.util.spec_from_file_location("ladder", PATH)
lad = importlib.util.module_from_spec(spec)
sys.path.insert(0, "/home/pino/absolver")
spec.loader.exec_module(lad)


def test_messages():
    m1 = lad.messages_for("resp_mod", "P", "R")
    m2 = lad.messages_for("user_only", "P", "R")
    assert [x["role"] for x in m1] == ["user", "assistant"], m1
    assert [x["role"] for x in m2] == ["user"], m2
    print("messages_for OK:", [x["role"] for x in m1], "|", [x["role"] for x in m2])


def test_aggregate_and_gates():
    rows = ([{"pair_id": f"h{i}", "expected": "Unsafe",
              "band": "flag" if i < 3 else "safe", "margin": 5.0 + i * 0.1}
             for i in range(10)]
            + [{"pair_id": f"b{i}", "expected": "Safe", "band": "safe",
                "margin": -4.0 - i * 0.1} for i in range(10)])
    st = lad.aggregate(rows)
    assert abs(st["auc"] - 1.0) < 1e-9, st["auc"]
    assert abs(st["flag_rate"] - 0.3) < 1e-9, st["flag_rate"]
    assert abs(st["benign_safe_rate"] - 1.0) < 1e-9, st["benign_safe_rate"]
    assert st["refit"]["harmful_recovered"] == 10, st["refit"]
    assert st["refit"]["benign_fp"] == 0, st["refit"]
    g = lad.gate_flags(types.SimpleNamespace(), st)
    assert g["discrimination_pass"] is True and g["benign_pass"] is True, g
    assert g["flag_rate_pass"] is False, g  # 30% flagged is a failed suppression
    print("aggregate/gates OK: AUC", st["auc"], "flag_rate", st["flag_rate"],
          "refit", st["refit"])

    # a constant-Safe reader: passes the two rate gates, must fail AUC
    flat = ([{"expected": "Unsafe", "band": "safe", "margin": 0.0} for _ in range(10)]
            + [{"expected": "Safe", "band": "safe", "margin": 0.0} for _ in range(10)])
    st2 = lad.aggregate(flat)
    g2 = lad.gate_flags(types.SimpleNamespace(), st2)
    assert g2["discrimination_pass"] is False, g2
    print("constant-Safe reader correctly fails the AUC gate:", st2["auc"])


class Lin(torch.nn.Module):
    def __init__(self, o, i):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(o, i))


class Attn(torch.nn.Module):
    def __init__(self, o, i):
        super().__init__()
        self.o_proj = Lin(o, i)


class Mlp(torch.nn.Module):
    def __init__(self, o, i):
        super().__init__()
        self.down_proj = Lin(o, i)


class Layer(torch.nn.Module):
    def __init__(self, o, i):
        super().__init__()
        self.self_attn = Attn(o, i)
        self.mlp = Mlp(o, i)


class Inner(torch.nn.Module):
    def __init__(self, n, o, i):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(o, i) for _ in range(n)])


class Dummy(torch.nn.Module):
    def __init__(self, n=3, o=16, i=16):
        super().__init__()
        self.model = Inner(n, o, i)
        self.device = torch.device("cpu")


def test_edit_alpha_semantics():
    rng = np.random.default_rng(7)
    d = rng.standard_normal(16)
    d /= np.linalg.norm(d)
    dirs = {f"d{i}": d for i in range(3)}
    dt = torch.tensor(d, dtype=torch.float32)
    for alpha in (0.75, 1.0, 1.25):
        m = Dummy()
        w0 = m.model.layers[1].self_attn.o_proj.weight.detach().clone()
        w0d = m.model.layers[1].mlp.down_proj.weight.detach().clone()
        before = float(dt @ w0.float() @ dt)
        before_d = float(dt @ w0d.float() @ dt)
        edited = lad.apply_edit(m, dirs, [1], "o_proj,down_proj", alpha)
        after = float(dt @ m.model.layers[1].self_attn.o_proj.weight.detach().float() @ dt)
        after_d = float(dt @ m.model.layers[1].mlp.down_proj.weight.detach().float() @ dt)
        assert abs(after - before * (1 - alpha)) < 1e-4, (alpha, before, after)
        assert abs(after_d - before_d * (1 - alpha)) < 1e-4, (alpha, before_d, after_d)
        print(f"apply_edit alpha={alpha}: edited {len(edited)} tensors, "
              f"d^T W d {before:+.4f} -> {after:+.4f} (={before * (1 - alpha):+.4f})")
        assert len(edited) == 2
    print("apply_edit OK: alpha is the fraction of the d-projection removed")


if __name__ == "__main__":
    test_messages()
    test_aggregate_and_gates()
    test_edit_alpha_semantics()
    print("ALL OFFLINE CHECKS PASSED")
