"""Rung 12 on Modal: project the write BEFORE the norm, not after it.

Rungs 10 and 11 applied the removal to the norm's OUTPUT, y = gamma * (x / rms).
Rung 12 applies it to the raw branch write x, along a = gamma * r, which is the
operator a weight-space abliteration actually implements: a linear map on x whose
null space contains a.

The two are not the same operator, and that is not a detail. Removing a direction
from y demands a change in x of order ||x|| once gamma is small, and the norm then
re-normalises what the removal produced. Every hook result so far says the same
thing from that side: refusal goes, coherence goes with it, and F1 could not map the
bake onto the hook edit because they were never the same edit.

Arms, same 12 prompts, same 4096 window, base reused from rung 10:

  ours-pre-1.0      our paired direction, a = gamma * r, 48 places
  heretic-pre-1.0   Heretic-Max u1 as r, a = gamma * r, 39 places
  rand-pre-a        seeded random r, a = gamma * r, 48 places (control for this endpoint)

Pre-registered reading:
  P1  any real direction reaches >= 8 ANSWER of 12 with <= 2 GARBLED  -> the norm side
      was the whole campaign's error, and the direction was never the problem.
  P2  real directions still garble but the random control does not     -> the refusal
      axis is entangled with competence at these writes; abliteration here is a
      trade, not a removal, and that is the campaign's answer.
  P3  random and real behave alike                                   -> the measurement
      cannot see direction specificity at this endpoint either; stop and rebuild the
      instrument before another rung.
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

app = modal.App("fenrir-lfm3-r12")

# tag -> (directions path in the volume, strength)
ARMS = [
    ("ours-pre-1.0", "/out/r9/r9-directions.pt", 1.0),
    ("heretic-pre-1.0", "/out/r11/dirs-heretic.pt", 1.0),
    ("rand-pre-a", "/out/r11/dirs-rand-20261114.pt", 1.0),
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
    os.makedirs("/out/r12", exist_ok=True)

    results = {}
    for tag, dirs, strength in ARMS:
        t0 = time.perf_counter()
        print(f"===== {tag} (where=pre) ...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, "-u", f"{_CAMP}/desktop/hookgate.py",
             "--config", "models/lfm3-1.7b-r8.yaml",
             "--dirs", dirs, "--position", "mean", "--where", "pre",
             "--strength", str(strength), "--only", "hook",
             "--base-from", "/out/r10/hookgate.json",
             "--out", f"/out/r12/hookgate-{tag}.json"],
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
