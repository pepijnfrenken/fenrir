"""Rung 20 on Modal: the two public builds' recovered axes, in the regime those builds work in.

Rungs 11 and 12 tested the recovered directions of Heretic-Max and rSalone — thinking-ON, at
the post-norm and pre-norm endpoints. Neither cleared the bar (heretic 6 ANSWER / 5 GARBLED,
rsalone 4 / 8). Then rung 16 found the instrument had never set the thinking regime, and both
public builds report their figures thinking-OFF. The recovered directions have never been
tested in the regime their source builds are scored in.

This rung does exactly that, with no other change: same 12 held-out prompts, same greedy 4096
window, same pre-norm endpoint, thinking OFF.

  heretic-pre-1.0-nothink    Heretic-Max's recovered axis, unit dose
  heretic-pre-2.0-nothink    ... at twice that
  rsalone-pre-0.5-nothink    rSalone's recovered axes, which carry real write energy
                             (rho 0.056 mean, 0.251 at L0) instead of 0.029
  rsalone-pre-1.0-nothink
  rsalone-pre-2.0-nothink

Base arm is reused from rung 16 (`base-nothink`: 12/12 REFUSE, mean 557 chars, agreement 12/12).

Pre-registered:
  P1  any arm reaches ANSWER >= 8/12 with GARBLED <= 2  ->  the axis that works is one of the
      published ones, and the campaign can build its own checkpoint from that axis plus its own
      per-place optimisation. That is the route to a shipped model, and it says plainly that the
      paired harvest was the weak object.
  P2  every arm lands on the same refusal-for-garble curve as our own direction  ->  the harvest
      is not the variable and neither is the direction's provenance; the campaign's own recipe
      is closed and the working builds differ in something neither card states.
  P3  rsalone's coupled axes beat heretic's near-generic ones at matched dose  ->  write-space
      coupling (rho) predicts edit efficiency, and that becomes the selection criterion for any
      future harvest.

Cost ceiling: one L4, five arms, 60 minutes.
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

app = modal.App("fenrir-lfm3-r20")

ARMS = [
    ("heretic-pre-1.0-nothink", "/absolver/pubdirs/dirs-heretic.pt", ["--strength", "1.0"]),
    ("heretic-pre-2.0-nothink", "/absolver/pubdirs/dirs-heretic.pt", ["--strength", "2.0"]),
    ("rsalone-pre-0.5-nothink", "/absolver/pubdirs/dirs-rsalone.pt", ["--strength", "0.5"]),
    ("rsalone-pre-1.0-nothink", "/absolver/pubdirs/dirs-rsalone.pt", ["--strength", "1.0"]),
    ("rsalone-pre-2.0-nothink", "/absolver/pubdirs/dirs-rsalone.pt", ["--strength", "2.0"]),
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
        .add_local_file(str(_PROJECT / ".forensics" / "r11" / "dirs-heretic.pt"),
                        "/absolver/pubdirs/dirs-heretic.pt")
        .add_local_file(str(_PROJECT / ".forensics" / "r11" / "dirs-rsalone.pt"),
                        "/absolver/pubdirs/dirs-rsalone.pt")
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
    os.makedirs("/out/r20", exist_ok=True)
    print(f"container up; arms={[a for a, _, _ in ARMS]}", flush=True)

    results = {}
    for tag, dirs, extra in ARMS:
        t0 = time.perf_counter()
        print(f"===== {tag} ...", flush=True)
        cmd = [sys.executable, "-u", f"{_CAMP}/desktop/hookgate.py",
               "--config", "models/lfm3-1.7b-r8.yaml",
               "--dirs", dirs, "--position", "mean", "--where", "pre",
               "--only", "hook", "--no-think",
               "--out", f"/out/r20/hookgate-{tag}.json"] + extra
        proc = subprocess.Popen(cmd, cwd="/absolver", env=env, stdout=subprocess.PIPE,
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
    print(json.dumps(run_rung.remote(), indent=1))
