"""Rung 11b on Modal: the control rung 11 was missing.

Rung 11 hooked four different directions at all 48 post-norm places and every arm
rambled (mean 7.6k-9.3k chars against a base mean of 1,365). An effect that appears
for our direction and for both public directions equally is not evidence about any
direction. The missing control is a direction with no relation to refusal at all.

Two arms, same prompts, same window, same base:

  rand-20261114-a   seeded gaussian unit vector per place
  rand-20261114-b   second seed, so a single lucky draw cannot carry the story

The question this answers is not "does the model garble" but "does removal of an
arbitrary direction at these 48 places garble it too". If the random arms ramble and
lose refusal as well, then the hook endpoint is measuring the fragility of the
sandwich-norm residual stream, not a refusal axis, and no direction comparison at
this endpoint can be read as abliteration evidence.

Pre-registered reading:
  R1  random arms garbled and refusal-free  -> the endpoint is the confound; stop
      comparing directions here and go back to weight space with a matched-rel_change
      bake target taken from the public builds.
  R2  random arms keep refusals             -> direction specificity is real and rung
      11's four arms get read for their differences.
  R3  mixed across the two seeds            -> the measurement is too noisy at n=12 to
      carry any claim; say so and raise n before another arm.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

_CAMP = "campaigns/liquidai-lfm3-1.7b"

if modal.is_local():
    _PROJECT = Path(__file__).resolve().parents[2]
else:
    _PROJECT = Path("/absolver")

app = modal.App("fenrir-lfm3-r11b")

SEEDS = [20261114, 20261115]

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

    import torch

    env = dict(os.environ)
    env.update({"PYTHONPATH": "/absolver", "ABSOLVER_CAMPAIGNS_ROOT": "/out",
                "HF_HOME": "/out/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    os.makedirs("/out/r11", exist_ok=True)

    # random unit direction per place, 2048-dim, seeded
    places = []
    for li in range(24):
        for pl in ("operator", "ffn"):
            places.append(f"L{li}.{pl}")
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        dirs = {}
        for p in places:
            v = torch.randn(2048, generator=g, dtype=torch.float32)
            dirs[p] = v / v.norm()
        path = f"/out/r11/dirs-rand-{seed}.pt"
        torch.save({"dirs": {"mean": dirs, "last": dirs},
                    "model_id": "random-control", "seed": seed}, path)
        print(f"wrote {path}", flush=True)

    results = {}
    for seed in SEEDS:
        tag = f"rand-{seed}"
        t0 = time.perf_counter()
        print(f"===== {tag} ...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, "-u", f"{_CAMP}/desktop/hookgate.py",
             "--config", "models/lfm3-1.7b-r8.yaml",
             "--dirs", f"/out/r11/dirs-rand-{seed}.pt", "--position", "mean",
             "--strength", "1.0", "--only", "hook",
             "--base-from", "/out/r10/hookgate.json",
             "--out", f"/out/r11b/hookgate-{tag}.json"],
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
