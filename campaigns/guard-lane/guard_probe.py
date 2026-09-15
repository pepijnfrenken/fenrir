#!/usr/bin/env python3
"""Guard lane — Step 2 PROBE: harvest the guard's verdict-position activations,
distill the harm direction, and steer-test causality before any weight edit.

The guard decides the verdict at ONE position: the last token of "...Safety:"
(the next token is " Unsafe"/" Safe"). That state is the abliteration-shaped
object for a classifier — the analog of the refusal direction in chat targets.

Subcommands:
  harvest  — one teacher-forced forward per pair (text + "Safety:"); hidden
             states at the decision position saved per layer (npz). TRAIN pairs
             build the direction; TUNE pairs validate it; TEST stays untouched
             until the final gates.
  distill  — diff_means direction per layer over TRAIN pairs; separation (norm
             of mean contrast) + sign-accuracy of the direction on TUNE
             (generalization check); writes directions npz + layer profile.
  steer    — causality test (no weight edit):
             (a) decision-position steering: add alpha*d at the last position
                 and read the verdict margin; the flip curve per alpha;
             (b) generation steering: hook the layer during generation and
                 record whether the EMITTED verdict flips.
             No flip anywhere -> the direction is a content proxy, not the
             decision circuit. Stop there (that is the honest finding).

Usage:
  python campaigns/guard-lane/guard_probe.py harvest --pairs campaigns/guard-lane/data/guard_pairs_lfm-abl_n40.jsonl
  python campaigns/guard-lane/guard_probe.py distill
  python campaigns/guard-lane/guard_probe.py steer --layer 14
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
from guard_readout import parse_verdict, resolve_verdict_tokens  # noqa: E402

RESULTS = HERE / "results"
CONFIG = "models/qwen3guard-0.6b.yaml"


def log(msg: str) -> None:
    print(f"[guard-probe] {msg}", flush=True)


def load_guard(cfg):
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
    return tok, model


def load_pairs(path: str, splits=("train", "tune")):
    pairs = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [p for p in pairs if p.get("split") in splits]


def decision_text(tok, prompt: str, response: str) -> str:
    """The guard input up to (and including) 'Safety:' — the decision state."""
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "Safety:"


def verdict_margin_from_logits(logits, tok_ids) -> float:
    s_unsafe = max(float(logits[i]) for i in tok_ids["unsafe"])
    s_safe = max(float(logits[i]) for i in tok_ids["safe"])
    return s_unsafe - s_safe


# --------------------------------------------------------------------- #
# harvest
# --------------------------------------------------------------------- #

def cmd_harvest(args) -> None:
    cfg = load_config(args.config)
    tok, model = load_guard(cfg)
    pairs = load_pairs(args.pairs)
    log(f"{len(pairs)} pairs (train+tune) for harvest")
    layer_acts: dict[int, list] = {}
    labels, pair_ids, splits = [], [], []
    t0 = time.time()
    for k, p in enumerate(pairs):
        text = decision_text(tok, p["prompt"], p["response"])
        inp = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states  # (embeddings, layer_0_out, ..., layer_L-1_out)
        for l in range(1, len(hs)):
            layer_acts.setdefault(l - 1, []).append(hs[l][0, -1, :].float().cpu().numpy())
        labels.append(1 if p.get("expected") == "Unsafe" else 0)
        pair_ids.append(p.get("pair_id"))
        splits.append(p.get("split"))
        if (k + 1) % 20 == 0:
            log(f"harvested {k+1}/{len(pairs)} ({time.time()-t0:.0f}s)")
    n_layers = len(layer_acts)
    out_path = RESULTS / f"probe_{args.tag}.npz"
    np.savez(out_path,
             labels=np.array(labels),
             pair_ids=np.array(pair_ids, dtype=object),
             splits=np.array(splits, dtype=object),
             n_layers=np.array([n_layers]),
             **{f"h{l}": np.stack(v) for l, v in layer_acts.items()})
    log(f"wrote {out_path} ({n_layers} layers x {len(pairs)} pairs x {layer_acts[0][0].shape[0]} dims)")
    # quick profile: resid norm per layer
    for l in sorted(layer_acts):
        norms = np.linalg.norm(np.stack(layer_acts[l]), axis=1)
        log(f"  layer {l:2d}: mean resid norm {norms.mean():.1f}")


# --------------------------------------------------------------------- #
# distill
# --------------------------------------------------------------------- #

def cmd_distill(args) -> None:
    data = np.load(RESULTS / f"probe_{args.tag}.npz", allow_pickle=True)
    n_layers = int(data["n_layers"][0])
    labels = data["labels"]
    splits = np.array([str(s) for s in data["splits"]])
    train = splits == "train"
    tune = splits == "tune"
    profile, directions = [], {}
    for l in range(n_layers):
        h = data[f"h{l}"]
        d = h[train & (labels == 1)].mean(axis=0) - h[train & (labels == 0)].mean(axis=0)
        sep = float(np.linalg.norm(d))
        d_unit = d / (sep + 1e-12)
        directions[l] = d_unit
        # generalization: sign separation on TUNE (threshold = midpoint of train proj means)
        proj_tr = h[train] @ d_unit
        thr = 0.5 * (proj_tr[labels[train] == 1].mean() + proj_tr[labels[train] == 0].mean())
        proj_tu = h[tune] @ d_unit
        pred = (proj_tu > thr).astype(int)
        acc = float((pred == labels[tune]).mean()) if tune.sum() else float("nan")
        profile.append({"layer": l, "separation": round(sep, 3),
                        "tune_sign_acc": round(acc, 4),
                        "resid_norm": round(float(np.linalg.norm(h, axis=1).mean()), 1)})
    out_path = RESULTS / f"directions_{args.tag}.npz"
    np.savez(out_path, **{f"d{l}": v for l, v in directions.items()},
             n_layers=np.array([n_layers]))
    (RESULTS / f"directions_{args.tag}_profile.json").write_text(json.dumps(profile, indent=2))
    best = sorted(profile, key=lambda r: (-r["tune_sign_acc"], -r["separation"]))[:6]
    log("top layers by tune sign-accuracy / separation:")
    for r in best:
        log(f"  layer {r['layer']:2d}: tune_acc={r['tune_sign_acc']:.3f} sep={r['separation']:.2f} norm={r['resid_norm']}")
    log(f"wrote {out_path} + profile")


# --------------------------------------------------------------------- #
# steer
# --------------------------------------------------------------------- #

def _hook_all_positions(alpha: float, d: torch.Tensor):
    def fn(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
            return (h + (alpha * d).to(h.dtype),) + tuple(output[1:])
        h = output
        return h + (alpha * d).to(h.dtype)
    return fn


def _hook_last_position(alpha: float, d: torch.Tensor):
    def fn(module, args, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[:, -1, :] = h[:, -1, :] + (alpha * d).to(h.dtype)
            return (h,) + tuple(output[1:])
        h = output.clone()
        h[:, -1, :] = h[:, -1, :] + (alpha * d).to(h.dtype)
        return h
    return fn


def cmd_steer(args) -> None:
    cfg = load_config(args.config)
    tok, model = load_guard(cfg)
    tok_ids = resolve_verdict_tokens(tok, getattr(cfg, "verdict_tokens", None) or None)
    data = np.load(RESULTS / f"probe_{args.tag}.npz", allow_pickle=True)
    dirs = np.load(RESULTS / f"directions_{args.tag}.npz")
    splits = np.array([str(s) for s in data["splits"]])
    labels = data["labels"]
    tune_idx = [i for i, s in enumerate(splits) if s == "tune"]
    pairs = load_pairs(args.pairs, splits=("tune",))
    assert len(pairs) == len(tune_idx), (len(pairs), len(tune_idx))
    tune_labels = labels[tune_idx]  # aligned with `pairs` (same file order)

    layer = args.layer
    d = torch.tensor(dirs[f"d{layer}"], dtype=torch.float32).to(model.device)
    scale = float(np.linalg.norm(data[f"h{layer}"], axis=1).mean())
    alphas = [a * scale for a in args.alphas]
    log(f"layer {layer} · direction scale (mean resid norm) {scale:.1f} · alphas {[round(a,1) for a in alphas]}")

    layer_mod = model.model.layers[layer]

    # (a) decision-position steering: margin curve on tune pairs
    curve = []
    harmful = [j for j in range(len(pairs)) if tune_labels[j] == 1]
    benign = [j for j in range(len(pairs)) if tune_labels[j] == 0]
    for alpha in alphas:
        handle = layer_mod.register_forward_hook(_hook_last_position(alpha, d))
        h_margins, b_margins = [], []
        try:
            for j, p in enumerate(pairs):
                text = decision_text(tok, p["prompt"], p["response"])
                inp = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_len).to(model.device)
                with torch.no_grad():
                    out = model(**inp)
                m = verdict_margin_from_logits(out.logits[0, -1].float(), tok_ids)
                (h_margins if tune_labels[j] == 1 else b_margins).append(m)
        finally:
            handle.remove()
        row = {"alpha": round(alpha, 2),
               "harmful_margin_mean": round(float(np.mean(h_margins)), 3),
               "benign_margin_mean": round(float(np.mean(b_margins)), 3),
               "harmful_flipped": int(sum(1 for m in h_margins if m < 0)),
               "harmful_n": len(h_margins)}
        curve.append(row)
        log(f"  (a) alpha={row['alpha']:>8}: harmful margin {row['harmful_margin_mean']:>7} "
            f"(flips {row['harmful_flipped']}/{row['harmful_n']}), benign {row['benign_margin_mean']:>7}")

    # (b) generation steering: does the EMITTED verdict flip?
    gen_rows = []
    hpairs = [pairs[j] for j in harmful]
    for alpha in [a for a in alphas if a <= 0] or [0.0]:
        handle = layer_mod.register_forward_hook(_hook_all_positions(alpha, d))
        flags = unparsed = 0
        raws = []
        try:
            for p in hpairs:
                messages = [{"role": "user", "content": p["prompt"]},
                            {"role": "assistant", "content": p["response"]}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_len).to(model.device)
                with torch.no_grad():
                    g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                raw = tok.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                j = parse_verdict(raw, "qwen3guard")
                flags += int(j.band == "flag")
                unparsed += int(j.band == "unparsed")
                if len(raws) < 3:
                    raws.append(raw.replace("\n", " | ")[:90])
        finally:
            handle.remove()
        gen_rows.append({"alpha": round(alpha, 2), "flagged": flags, "n": len(hpairs),
                         "unparsed": unparsed, "sample_raws": raws})
        log(f"  (b) alpha={alpha/scale:+.2f}x·scale: flagged {flags}/{len(hpairs)} (unparsed {unparsed})")

    out = {"layer": layer, "scale": scale, "decision_curve": curve, "generation_curve": gen_rows}
    (RESULTS / f"steer_{args.tag}_l{layer}.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {RESULTS / f'steer_{args.tag}_l{layer}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("harvest", "steer"):
        s = sub.add_parser(name)
        s.add_argument("--pairs", default="campaigns/guard-lane/data/guard_pairs_lfm-abl_n40.jsonl")
        s.add_argument("--config", default=CONFIG)
        s.add_argument("--tag", default="guard")
        s.add_argument("--max-len", type=int, default=1024)
        if name == "steer":
            s.add_argument("--layer", type=int, required=True)
            s.add_argument("--alphas", type=float, nargs="+",
                           default=[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0])
    d = sub.add_parser("distill")
    d.add_argument("--tag", default="guard")
    args = ap.parse_args()
    if args.cmd == "harvest":
        cmd_harvest(args)
    elif args.cmd == "distill":
        cmd_distill(args)
    else:
        cmd_steer(args)


if __name__ == "__main__":
    main()
