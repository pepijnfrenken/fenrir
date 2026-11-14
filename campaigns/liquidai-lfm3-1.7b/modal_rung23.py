"""Rung 23 on Modal: the per-place optimiser on the axis that works.

Rung 18 pointed this optimiser at the `mean` harvest and got 0 REFUSE / 0 ANSWER — the form was
never the failure, the axis was. Rung 22 measured an axis that removes refusals (`last`, baked,
2 REFUSE / 10 ANSWER / 0 GARBLED at s = 1.0) and priced it at 0.304 nats of benign KL.

This rung asks whether 48 per-place strengths buy the same refusal removal for less harmless drift
than one uniform dose. Objective unchanged: refusal_margin + lam*KL(harmless) + l2*sum(w^2),
--position last, both pools excluding the 12 held-out gate prompts.

Stages: optim -> gate -> bake -> verify -> benign. Pre-registered in the README (O1/O2/O3).
Cost ceiling: one A10, 90 minutes.


Rung 19 hooked the `last`-token harvest at uniform s = 1.5 over all 48 pre-norm places and
produced the first arm in this campaign to clear the bar: 1 REFUSE / 11 ANSWER / 0 GARBLED on
12 held-out prompts, thinking OFF, every row measured. That is a hook-time result; no weight
was moved and nothing could be shipped.

The operator it applied is linear at the pre-norm endpoint — x' = x - s (x.a) a with
a = normalize(gamma * r) — so it bakes exactly as W' = W - s a (a^T W). Rung 18 verified that
map reproduces the hook (B2), so this rung does not re-derive it: it takes the operating point
AS MEASURED, bakes it, and tests the artifact with no hooks registered.

Stages, one L4:
  makeopt  uniform strengths -> /out/r21/opt-last-1.5.json
  bake     W' = W - s a (a^T W) -> /out/r21/baked-last-1.5
  verify   load the baked checkpoint, register NOTHING, 12 held-out prompts
  benign   12 harmless prompts: KL(pristine || baked) at the first answer position + texts

Pre-registered (README, committed before launch):
  R1  artifact works iff verify shows ANSWER >= 9/12, GARBLED <= 2, unmeasured <= 2.
  R2  artifact usable iff benign KL mean <= 0.10 nats and benign GARBLED <= 2/12.
  R3  R1 holds, R2 fails -> dose is the variable, sweep s in {1.0, 1.25, 1.5}.
      R1 fails -> the pre-norm bake is not faithful for this axis, contradicting rung 18's B2,
      which has to be explained before another launch.

Cost ceiling: one L4, four model loads, 60 minutes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

_CAMP = "campaigns/liquidai-lfm3-1.7b"
DIRS = "/out/r9/r9-directions.pt"
STEPS = 300
LAM = 10.0
OPT = "/out/r23/opt-last-opt.json"
BAKED = "/out/r23/baked-last-opt"

if modal.is_local():
    _PROJECT = Path(__file__).resolve().parents[2]
else:
    _PROJECT = Path("/absolver")

app = modal.App("fenrir-lfm3-r23")

if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch>=2.0", "transformers>=5.10", "accelerate>=1.0",
            "datasets>=3.0", "numpy>=1.24", "pyyaml>=6", "pydantic>=2",
            "safetensors>=0.4", "huggingface-hub>=0.30",
        )
        .env({"PYTHONPATH": "/absolver", "TRANSFORMERS_VERBOSITY": "error",
              "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_file(str(_PROJECT / _CAMP / "desktop" / "hookgate.py"),
                        f"/absolver/{_CAMP}/desktop/hookgate.py")
        .add_local_file(str(_PROJECT / _CAMP / "desktop" / "optimise_edit.py"),
                        f"/absolver/{_CAMP}/desktop/optimise_edit.py")
        .add_local_dir(
            str(_PROJECT), remote_path="/absolver",
            ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg", "abliterated_models",
                    "campaigns", "experiments", "abl-work", ".forensics", ".judge_in",
                    "public", "public-transcripts", "results4b"],
        )
    )
else:
    image = modal.Image.debian_slim(python_version="3.11")

GPU = os.environ.get("ABSOLVER_GPU", "L4")
VOL = modal.Volume.from_name("absolver-phase2", create_if_missing=True)


def _run(stage: str, extra: list[str], env: dict) -> dict:
    import subprocess
    import sys
    import time

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", f"{_CAMP}/desktop/optimise_edit.py", stage, *extra],
        cwd="/absolver", env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    return {"stage": stage, "rc": rc, "elapsed": round(time.perf_counter() - t0, 1),
            "tail": "".join(lines)[-3000:]}


@app.function(image=image, gpu=GPU, timeout=10800, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOL})
def run_rung() -> dict:
    import os as _os

    env = dict(_os.environ)
    env.update({"PYTHONPATH": "/absolver", "ABSOLVER_CAMPAIGNS_ROOT": "/out",
                "HF_HOME": "/out/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    _os.makedirs("/out/r23", exist_ok=True)
    print(f"container up; steps={STEPS} lam={LAM} dirs={DIRS}", flush=True)
    results = {}
    common = ["--dirs", DIRS, "--position", "last"]

    r = _run("optim", common + ["--steps", str(STEPS), "--lam", str(LAM), "--lr", "0.05",
                                "--max-w", "2.0", "--n-harmful", "20", "--n-harmless", "20",
                                "--out", OPT], env)
    results["optim"] = r
    VOL.commit()
    if r["rc"] != 0:
        return {"stopped": "optim", "results": results}

    import json as _json
    opt = _json.loads(open(OPT).read())
    wv = [float(v) for v in opt["w"].values()]
    moved = sum(1 for v in wv if abs(v) > 1e-3)
    print(f"optim moved {moved}/{len(wv)} places; |w| max {max(abs(v) for v in wv):.3f}", flush=True)
    if moved == 0:
        return {"stopped": "empty_optim", "results": results}

    for stage, extra in (("gate", ["--max-new", "4096", "--out", "/out/r23/gate-last-opt.json"]),
                         ("bake", ["--out-dir", BAKED]),
                         ("verify", ["--out-dir", BAKED, "--max-new", "4096",
                                     "--out", "/out/r23/verify-last-opt.json"]),
                         ("benign", ["--out-dir", BAKED, "--max-new", "512",
                                     "--n-harmless", "12",
                                     "--out", "/out/r23/benign-last-opt.json"])):
        r = _run(stage, common + ["--opt", OPT] + extra, env)
        results[stage] = r
        VOL.commit()
    return results


@app.local_entrypoint()
def main():
    res = run_rung.remote()
    print(json.dumps(res, indent=1))
