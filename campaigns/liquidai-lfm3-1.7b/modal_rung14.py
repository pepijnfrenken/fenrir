"""Rung 14 on Modal: the budget hypothesis, tested at the endpoint that matters.

Rung 12 removed the last algebraic objection. Pre-norm projection along a = gamma * r
is the operator a weight edit implements, and at strength 1.0 it produces 0 REFUSE /
3 ANSWER / 9 GARBLED against base 1,365 chars, mean 10,634. Random removal is inert
(24 of 24 REFUSE). So full removal is destructive and specific.

What is left is dose. The published builds move weights by 2.5-3.1% (Heretic-Max) and
0.5-17.3% per place (rSalone); a full rank-1 projection is 30-70% of the weight norm at
those places. Their published KLs are 0.0040 and 0.0013. That is an order of magnitude
smaller intervention than any strength this campaign has judged so far except rungs 5
and 6, and those two were measured at the post-norm endpoint with a keyword detector
that could not read LFM3's refusal prose.

Rung 14 tests the budget hypothesis with the right endpoint and the right instrument:
pre-norm projection, strengths that correspond to a small fraction of the write, every
arm blind-judged.

  ours-pre-0.05
  ours-pre-0.10
  ours-pre-0.20
  rand-pre-0.10   (dose-matched control: a random direction at the same strength)

Pre-registered reading:
  B1  any real arm reaches >= 8 ANSWER of 12 with <= 2 GARBLED at a strength where the
      random control leaves refusals in place -> a budgeted projection works and the
      discriminator was dose all along. Campaign turns positive and names the budget.
  B2  no real arm reaches that bar at any strength, refusals survive below it and
      competence dies above it -> there is no operating point. The refusal axis is not
      separable by projection on this architecture and the campaign closes negative
      with a measured trade curve instead of an opinion.
  B3  the random control also drops refusals at 0.10 -> direction specificity does not
      hold at this endpoint after all, and rung 11b's reading does not transfer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

_CAMP = "campaigns/liquidai-lfm3-1.7b"

if modal.is_local():
    _PROJECT = Path(__file__).resolve().parents[2]
else:
    _PROJECT = Path("/absolver")

app = modal.App("fenrir-lfm3-r14")

ARMS = [
    ("ours-pre-0.05", "/out/r9/r9-directions.pt", 0.05),
    ("ours-pre-0.10", "/out/r9/r9-directions.pt", 0.10),
    ("ours-pre-0.20", "/out/r9/r9-directions.pt", 0.20),
    ("rand-pre-0.10", "/out/r11/dirs-rand-20261114.pt", 0.10),
]

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


@app.function(image=image, gpu=GPU, timeout=14400, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOL})
def run_rung() -> dict:
    import subprocess
    import sys
    import time

    env = dict(os.environ)
    env.update({"PYTHONPATH": "/absolver", "ABSOLVER_CAMPAIGNS_ROOT": "/out",
                "HF_HOME": "/out/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    os.makedirs("/out/r14", exist_ok=True)

    results = {}
    for tag, dirs, strength in ARMS:
        t0 = time.perf_counter()
        print(f"===== {tag} (where=pre, s={strength}) ...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, "-u", f"{_CAMP}/desktop/hookgate.py",
             "--config", "models/lfm3-1.7b-r8.yaml",
             "--dirs", dirs, "--position", "mean", "--where", "pre",
             "--strength", str(strength), "--only", "hook",
             "--base-from", "/out/r10/hookgate.json",
             "--out", f"/out/r14/hookgate-{tag}.json"],
            cwd="/absolver", env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(f"[{tag}] {line}", end="", flush=True)
        rc = proc.wait()
        results[tag] = {"rc": rc, "elapsed": round(time.perf_counter() - t0, 1)}
        print(f"===== {tag} rc={rc} in {results[tag]['elapsed']}s", flush=True)
        VOL.commit()
    return results


@app.local_entrypoint()
def main():
    print(run_rung.remote())
