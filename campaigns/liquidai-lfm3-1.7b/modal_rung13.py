"""Rung 13 on Modal: remove the axis during the prompt pass only.

Rung 12 settled the algebra question. Projecting the raw branch write (a = gamma*r)
is the operator a weight edit implements, and at strength 1.0 it does exactly what the
post-norm projection did: 0 REFUSE / 3 ANSWER / 9 GARBLED of 12, mean 10,634 chars
against base 1,365. So the side of the norm was not the campaign's error. Full removal
of a refusal-aligned axis destroys usable generation wherever it is applied.

That leaves one way for the axis to be removable without the model going with it:
separate the two in time instead of in space. Refusal is decided while the prompt is
read. Competence is what generation needs after that. `--prefill-only` edits the
prompt pass and leaves every decode step untouched, using seq_len as the discriminator
(a prompt pass presents prompt_len, a decode step presents 1).

rSalone's card reports prompt-side directions beating thinking-side ones, which is the
same idea from their side of the work. This is the first rung that acts on it.

Arms, same 12 prompts, same 4096 window, base reused from rung 10:

  ours-pre-prompt    our paired direction, 48 places, prompt pass only
  heretic-pre-prompt Heretic-Max u1, 39 places, prompt pass only
  rand-pre-prompt    seeded random, 48 places, prompt pass only (endpoint control)

Pre-registered reading:
  T1  any real direction reaches >= 8 ANSWER of 12 with <= 2 GARBLED and the random
      arm does not -> the axis is separable in time. That is the campaign's positive
      finding: a tactic, not a checkpoint.
  T2  real directions still garble -> the axis is entangled with competence in the
      prompt pass too, and no projection formulation of abliteration works on this
      architecture. Campaign closes negative, with the mechanism named.
  T3  refusals survive in every arm -> the refusal decision is not made during prefill
      at these writes; the prompt-pass hypothesis is dead and the axis lives all the
      way through generation.
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

app = modal.App("fenrir-lfm3-r13")

# tag -> (directions path in the volume, strength)
ARMS = [
    ("ours-prompt-1.0", "/out/r9/r9-directions.pt", 1.0),
    ("heretic-prompt-1.0", "/out/r11/dirs-heretic.pt", 1.0),
    ("rand-prompt-a", "/out/r11/dirs-rand-20261114.pt", 1.0),
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
    os.makedirs("/out/r13", exist_ok=True)

    # Instrument check first: one prompt, short window, and read the tally back.
    # If the prompt/decode split does not actually happen, every arm below is void.
    print("===== smoke: verify the prompt/decode split", flush=True)
    smoke = subprocess.run(
        [sys.executable, "-u", f"{_CAMP}/desktop/hookgate.py",
         "--config", "models/lfm3-1.7b-r8.yaml",
         "--dirs", "/out/r9/r9-directions.pt", "--position", "mean",
         "--where", "pre", "--prefill-only", "--limit", "1", "--max-new", "96",
         "--only", "hook", "--out", "/out/r13/smoke.json"],
        cwd="/absolver", env=env, capture_output=True, text=True)
    print(smoke.stdout[-2000:], smoke.stderr[-2000:], flush=True)
    tally = json.loads(Path("/out/r13/smoke.json").read_text()).get("hook_tally") or {}
    places = 48
    ok = (tally.get("prompt_passes", 0) == places
          and tally.get("decode_passes", 0) > 0
          and tally.get("edited", 0) == tally.get("prompt_passes", -1))
    print(f"smoke tally {tally} -> ok={ok}", flush=True)
    if not ok:
        return {"smoke_failed": True, "tally": tally, "stdout": smoke.stdout[-2000:]}

    results = {}
    for tag, dirs, strength in ARMS:
        t0 = time.perf_counter()
        print(f"===== {tag} (where=pre, prefill_only) ...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, "-u", f"{_CAMP}/desktop/hookgate.py",
             "--config", "models/lfm3-1.7b-r8.yaml",
             "--dirs", dirs, "--position", "mean", "--where", "pre",
             "--prefill-only", "--strength", str(strength), "--only", "hook",
             "--base-from", "/out/r10/hookgate.json",
             "--out", f"/out/r13/hookgate-{tag}.json"],
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
