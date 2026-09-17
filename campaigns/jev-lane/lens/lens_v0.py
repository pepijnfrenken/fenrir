#!/usr/bin/env python3
"""jev-lane / lens v0: render a model's hidden state as TEXT for Jev.

Two modes:
  --build-items : build items_v0.json from the repo prompt sources (local only)
  (default)     : capture + logit-lens + stage for Jev

Capture: state read at the LAST PROMPT TOKEN (right before the first answer
token -> no answer text present -> no trivial leakage), four depths.

Lens: project the residual stream through the model's own final norm + lm_head
and keep the top-k tokens per depth ("what this depth would say if it had to
speak"). The norm+head path is self-checked against the model's own
last-position logits before anything is trusted.

Outputs are LOCAL (out/ is gitignored) — preview Interfaces ToS §1(v).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ITEMS_DEFAULT = HERE / "items_v0.json"

MODEL_DEFAULT = "LiquidAI/LFM2.5-1.2B-Instruct"
TAG_DEFAULT = "orig"
REFUSAL_EVAL = Path("/home/pino/absolver/campaigns/guard-lane/data/refusal_eval.jsonl")
TASKS_DIR = Path("/home/pino/projects/if-ablation/data/tasks")
N_PER_COND = 8
GEN_TOKENS = 20
TOPK = 12


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_items() -> list[dict]:
    items: list[dict] = []
    rows = load_jsonl(REFUSAL_EVAL)
    harmful = [r["prompt"] for r in rows if r.get("label") == 1][:N_PER_COND]
    benign = [r["prompt"] for r in rows if r.get("label") == 0][:N_PER_COND]
    for i, p in enumerate(harmful):
        items.append({"id": f"R{i}", "condition": "refuse", "prompt": p})
    for i, p in enumerate(benign):
        items.append({"id": f"A{i}", "condition": "answer", "prompt": p})
    quotas = [("word_count", 2), ("char_count", 2), ("term_exclusion", 2),
              ("term_inclusion", 1), ("sentiment", 1)]
    ti = 0
    for task, n in quotas:
        seen: set[str] = set()
        got = 0
        for r in load_jsonl(TASKS_DIR / f"{task}.jsonl"):
            p = r["prompt"].strip()
            if p in seen:
                continue
            seen.add(p)
            items.append({"id": f"T{ti}", "condition": "task", "prompt": p, "task": task})
            ti += 1
            got += 1
            if got >= n:
                break
    return items


def clean_tok(tok, tid: int) -> str:
    s = tok.decode([tid])
    s = s.replace("\n", "\\n")
    if s.strip() == "":
        s = "∅" if s == "" else repr(s)[1:-1]
    return s[:24]


def cmd_build(args) -> None:
    items = build_items()
    Path(args.items).write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"wrote {args.items}: {len(items)} items")
    for it in items:
        print(f"  {it['id']:>3} {it['condition']:<6} {it['prompt'][:70]!r}")


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.items).read_text())
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 6)))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lens] {args.model} | tag={args.tag} | device={device} | "
          f"{len(items)} items", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = None
    for dt, name in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
            print(f"[lens] loaded as {name} in {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as e:  # pragma: no cover
            print(f"[lens] load as {name} failed: {type(e).__name__}: {e}", flush=True)
    if model is None:
        raise SystemExit("could not load model")
    model.to(device)
    model.eval()

    base = model.model
    print(f"[lens] base children: {[n for n, _ in base.named_children()]}", flush=True)
    L = len(base.layers)
    hs_idx = sorted({max(1, L // 4), L // 2, (3 * L) // 4, L})
    print(f"[lens] {L} blocks; reading at hs indices {hs_idx} "
          f"(depths {[round(100 * k / L) for k in hs_idx]}%)", flush=True)

    W = model.lm_head.weight.float()

    def lens_topk(h: torch.Tensor, norm_mod):
        hn = h if norm_mod is None else norm_mod(h.to(norm_mod.weight.dtype)).float()
        logits = hn @ W.T
        _v, idx = torch.topk(logits, TOPK)
        return [int(t) for t in idx]

    cands = [(n, getattr(base, n, None)) for n in ("norm", "final_norm", "embedding_norm")]
    cands.append(("identity", None))

    # -- self check ----------------------------------------------------------
    first = items[0]
    msgs = [{"role": "user", "content": first["prompt"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.inference_mode():
        out = model(torch.tensor([ids]).to(device), output_hidden_states=True, use_cache=False)
    pos = len(ids) - 1
    ref_ids = set(int(t) for t in torch.topk(out.logits[0, pos].float(), TOPK).indices)
    chosen = None
    for name, mod in cands:
        if name != "identity" and mod is None:
            continue
        tidx = lens_topk(out.hidden_states[L][0, pos].float(), mod)
        overlap = len(ref_ids & set(tidx))
        print(f"[lens] self-check '{name}': overlap {overlap}/{TOPK}", flush=True)
        if chosen is None and overlap >= TOPK - 2:
            chosen = (name, mod)
    if chosen is None:
        raise SystemExit("[lens] NO norm path matches model logits -- aborting")
    print(f"[lens] using final-norm path: {chosen[0]}", flush=True)
    chosen_norm = chosen[1]

    # -- capture -------------------------------------------------------------
    rows = []
    for it in items:
        msgs = [{"role": "user", "content": it["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids]).to(device)
        with torch.inference_mode():
            out = model(inp, output_hidden_states=True, use_cache=False)
            pos = len(ids) - 1
            layers_out = {}
            for k in hs_idx:
                tidx = lens_topk(out.hidden_states[k][0, pos].float(), chosen_norm)
                layers_out[str(k)] = [clean_tok(tok, t) for t in tidx]
            _lv, li = torch.topk(out.logits[0, pos].float(), 5)
            rec = {
                "id": it["id"], "condition": it["condition"], "task": it.get("task"),
                "prompt": it["prompt"], "n_tokens": len(ids), "layers": layers_out,
                "actual_top5": [clean_tok(tok, int(t)) for t in li],
                "raw_vec32_75pct": [round(float(x), 3) for x in
                                    out.hidden_states[hs_idx[-2]][0, pos].float()[:32].cpu()],
                "hs_idx": hs_idx, "n_blocks": L, "model": args.model, "tag": args.tag,
            }
            gen = model.generate(inp, max_new_tokens=GEN_TOKENS, do_sample=False)
            rec["gen_prefix"] = tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True)
        rows.append(rec)
        print(f"  [{len(rows)}/{len(items)}] {rec['id']}: gen={rec['gen_prefix'][:48]!r}",
              flush=True)

    out_path = OUT / f"lens_states_{args.tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[lens] wrote {out_path}", flush=True)

    for cond in ("refuse", "answer", "task"):
        r = next(x for x in rows if x["condition"] == cond)
        print(f"\n=== SANITY {r['id']} ({cond}): {r['prompt'][:90]!r}")
        print(f"    generated: {r['gen_prefix'][:110]!r}")
        for k in [str(x) for x in hs_idx]:
            print(f"    hs{k}: {' | '.join(r['layers'][k])}")
    print(f"\n[lens] done in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-items", action="store_true")
    ap.add_argument("--items", default=str(ITEMS_DEFAULT))
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--tag", default=TAG_DEFAULT)
    args = ap.parse_args()
    if args.build_items:
        cmd_build(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
