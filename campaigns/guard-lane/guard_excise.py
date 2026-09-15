#!/usr/bin/env python3
"""Guard lane — Step 2 EXCISE: project the harm direction out of the guard's
weights (o_proj / down_proj, optional lm_head input-side), save the ablated
guard, and read the guard gates on the held-out TEST split.

W[out, in] -= alpha * d (d^T W)   — d is the unit direction in resid space and
must have d.shape[0] == out (o_proj and down_proj both output the resid dim).
lm_head stores [vocab, d_model]: the direction lives on the INPUT side there, so
--include-lm-head removes W d d^T instead (its sensitivity to the direction).

Pre-edit discipline: run `guard_probe.py steer` FIRST; if steering cannot flip
the verdict, this script should not be launched (the direction is then a content
proxy, not the decision circuit).

Usage:
  python campaigns/guard-lane/guard_excise.py --layer 14 [--alpha 1.0] [--layers-weights o_proj,down_proj] [--include-lm-head]
  python campaigns/guard-lane/guard_excise.py --layer 14 --tag harmdir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from config import load_config  # noqa: E402
from gates import gate_flag_rate, gate_pass_rate_benign, guard_axis_status  # noqa: E402

RESULTS = HERE / "results"
CONFIG = "models/qwen3guard-0.6b.yaml"


def log(msg: str) -> None:
    print(f"[guard-excise] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--pairs", default="campaigns/guard-lane/data/guard_pairs_lfm-abl_n40.jsonl")
    ap.add_argument("--tag", default="guard")
    ap.add_argument("--layer", type=int, default=None, help="single layer (convenience)")
    ap.add_argument("--layers", default=None, help="comma-separated layer list (overrides --layer)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--weights", default="o_proj,down_proj")
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--lm-head-layer", type=int, default=27,
                    help="layer whose direction is used for the lm_head input-side removal")
    ap.add_argument("--out", default=None, help="output model dir (default under abliteration-local/models)")
    ap.add_argument("--no-save", action="store_true", help="evaluate only, do not write the checkpoint")
    args = ap.parse_args()

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else ([args.layer] if args.layer is not None else None))
    if not layers:
        sys.exit("pass --layer N or --layers a,b,c")
    lbl = "-".join(str(x) for x in layers)

    cfg = load_config(args.config)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(getattr(cfg, "dtype", "bfloat16"), torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype, device_map="cuda")
    model.eval()

    dirs = np.load(RESULTS / f"directions_{args.tag}.npz")
    log(f"editing layers {layers} · weights {args.weights} · alpha {args.alpha}")

    targets = [w.strip() for w in args.weights.split(",") if w.strip()]

    def _find_module(layer, name):
        """Resolve a weight name on a decoder layer across common nestings
        (layer.o_proj, layer.self_attn.o_proj, layer.mlp.down_proj, ...)."""
        for path in (name, f"self_attn.{name}", f"mlp.{name}"):
            mod = layer
            try:
                for part in path.split("."):
                    mod = getattr(mod, part)
                return mod
            except AttributeError:
                continue
        return None

    edited = []
    with torch.no_grad():
        for l in layers:
            d = torch.tensor(dirs[f"d{l}"], dtype=torch.float32).to(model.device)
            layer = model.model.layers[l]
            for name in targets:
                mod = _find_module(layer, name)
                if mod is None:
                    log(f"  WARNING: could not resolve {name!r} on layer {l}")
                    continue
                W = mod.weight
                W32 = W.float()
                proj = d @ W32                      # [in]
                W32 = W32 - args.alpha * torch.outer(d, proj)  # [out, in]
                W.copy_(W32.to(W.dtype))
                edited.append(f"L{l}.{name}")
        if args.include_lm_head:
            d_lm = torch.tensor(dirs[f"d{args.lm_head_layer}"], dtype=torch.float32).to(model.device)
            W = model.lm_head.weight
            W32 = W.float()
            W32 = W32 - args.alpha * torch.outer(W32 @ d_lm, d_lm)
            W.copy_(W32.to(W.dtype))
            edited.append(f"lm_head(input-side, d{args.lm_head_layer})")
    if not edited:
        sys.exit("no tensors edited — weight names could not be resolved on this architecture")
    log(f"edited {len(edited)} tensors: {edited[0]} … {edited[-1]}")

    # ---- evaluate on the held-out TEST split (in-memory, gates machinery)
    pairs = [json.loads(line) for line in open(args.pairs, encoding="utf-8")]
    test = [p for p in pairs if p.get("split") == "test"]
    transcript: list[dict] = []
    t0 = time.time()
    fr = gate_flag_rate(model, tok, cfg, test, transcript=transcript)
    pr = gate_pass_rate_benign(model, tok, cfg, test, transcript=transcript)
    axis = guard_axis_status(fr)
    log(f"TEST flag_rate       : {fr['value']} (passed {fr['passed']}) — {fr['detail']}")
    log(f"TEST pass_rate_benign: {pr['value']} (passed {pr['passed']}) — {pr['detail']}")
    log(f"guard axis status   : measurable={axis['measurable']} suspect={axis['instrument_suspect']} — {axis['detail']}")
    log(f"({time.time()-t0:.0f}s)")

    # ---- save the ablated checkpoint
    out_dir = Path(args.out) if args.out else Path("/home/pino/projects/abliteration-local/models") / f"qwen3guard-0.6b-abl-l{lbl}-a{args.alpha}"
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)
        tok.save_pretrained(out_dir)
        log(f"saved ablated guard -> {out_dir}")

    report = {
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "base_model": cfg.model_id, "layers": layers, "alpha": args.alpha,
        "weights": args.weights, "include_lm_head": bool(args.include_lm_head),
        "n_tensors_edited": len(edited), "out_dir": str(out_dir),
        "test_flag_rate": fr, "test_pass_rate_benign": pr, "guard_axis": axis,
        "per_pair": transcript,
    }
    rep_path = RESULTS / f"excise_{args.tag}_l{lbl}_a{args.alpha}.json"
    rep_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {rep_path}")


if __name__ == "__main__":
    main()
