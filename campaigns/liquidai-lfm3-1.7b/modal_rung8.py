"""LFM3-1.7B rung 8 on Modal: reproduce the published heretic recipe on OUR
weights, then gate it with OUR instruments at a window that clears thinking.

Why Modal: the desktop is off. The connector `connectors/harness_modal.py` only
runs `harness.abl` subcommands, and rung 8 needs two plain scripts
(reproduce_heretic.py, interference_check.py), so this lane has explicit stages
instead of string parsing. Same image conventions, same volume, same
ABSOLVER_CAMPAIGNS_ROOT contract.

Stages (each is reviewable on its own; run all in ONE container so the checkpoint
written by `edit` is visible to `gate`):
  dir           harvest 400 harmful / 400 harmless residuals at the first
                generated token, build the orthogonalised global direction,
                report separation and cos against our paired per-layer direction
                (F1)
  edit          apply the disclosed tent lambda profile to o_proj + w2 on layers
                11-23, row magnitudes restored -> checkpoint in the volume
  gate          harness.abl collect on that checkpoint, 12-prompt held-out split,
                4096-token window, interference on 20 harmless prompts (F2/F3)
  interference  first-token KL + PPL on the harmless pool for the public Heretic
                build AND our reproduction, on the same meter

Usage:
  modal run campaigns/liquidai-lfm3-1.7b/modal_rung8.py
  modal run campaigns/liquidai-lfm3-1.7b/modal_rung8.py --only gate
  ABSOLVER_GPU=L40S modal run campaigns/liquidai-lfm3-1.7b/modal_rung8.py
  modal volume get absolver-phase2 r8 /tmp/r8      # pull artifacts back
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

_CAMP = "campaigns/liquidai-lfm3-1.7b"

# Modal imports this module INSIDE the container (as /root/modal_rung8.py) to
# hydrate the function, so any path arithmetic on __file__ must be local-only.
# The image (and its add_local_* mounts) is likewise built only on the client.
if modal.is_local():
    _PROJECT = Path(__file__).resolve().parents[2]      # repo root
    _DIRS_PT = _PROJECT / _CAMP / "directions-chat-paired-t320.pt"
    assert _DIRS_PT.exists(), f"missing {_DIRS_PT} (needed for the F1 cosine)"
else:
    _PROJECT = Path("/absolver")
    _DIRS_PT = _PROJECT / _CAMP / "directions-chat-paired-t320.pt"

app = modal.App("fenrir-lfm3-r8")

if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch>=2.0",
            "transformers>=5.10",      # LFM3 remote code needs the 5.x API
            "accelerate>=1.0",
            "datasets>=3.0",           # mlabonne/harmless_alpaca + harmful_behaviors
            "numpy>=1.24",
            "pyyaml>=6",
            "pydantic>=2",
            "safetensors>=0.4",
            "huggingface-hub>=0.30",
        )
        .env({"PYTHONPATH": "/absolver", "TRANSFORMERS_VERBOSITY": "error",
              "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
        # the paired direction file lives under campaigns/, which the dir mount
        # ignores, so it is added explicitly (F1 needs it inside the container)
        .add_local_file(str(_DIRS_PT), f"/absolver/{_CAMP}/directions-chat-paired-t320.pt")
        # the dir mount excludes campaigns/ (it holds multi-GB checkpoints), so the
        # scripts this lane runs are added one by one at their real paths
        .add_local_file(str(_PROJECT / _CAMP / "desktop" / "reproduce_heretic.py"),
                        f"/absolver/{_CAMP}/desktop/reproduce_heretic.py")
        .add_local_file(str(_PROJECT / _CAMP / "desktop" / "interference_check.py"),
                        f"/absolver/{_CAMP}/desktop/interference_check.py")
        .add_local_dir(  # must be LAST build step
            str(_PROJECT),
            remote_path="/absolver",
            ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg", "abliterated_models",
                    "campaigns", "experiments", "abl-work", ".forensics", ".judge_in",
                    "public", "public-transcripts", "results4b"],
        )
    )
else:   # in-container: never built, only referenced by the hydrated function
    image = modal.Image.debian_slim(python_version="3.11")

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "10800"))

VOL = modal.Volume.from_name("absolver-phase2", create_if_missing=True)

STAGES = ("pristine", "dir", "edit", "gate", "interference")
# the pristine bundle is what certifies the refusal axis; without it the
# gate stage hard-fails BASELINE UNMEASURABLE for any --model-dir run
_PRISTINE = "/out/liquidai-lfm3-1.7b/collect-pristine/bundle.json"


def _run(argv: list[str], env: dict, cwd: str = "/absolver") -> dict:
    import subprocess
    import sys
    import time
    started = time.perf_counter()
    proc = subprocess.run([sys.executable] + argv, cwd=cwd, env=env,
                          capture_output=True, text=True)
    return {"argv": argv, "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:], "stderr": proc.stderr[-12000:],
            "elapsed_seconds": round(time.perf_counter() - started, 1)}


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOL})
def run_rung(stages: list[str] = list(STAGES)) -> dict:
    os.makedirs("/out/r8", exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "/absolver"
    env["ABSOLVER_CAMPAIGNS_ROOT"] = "/out"
    env["HF_HOME"] = "/out/hf"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    results = []
    for stage in stages:
        if stage == "pristine":
            if os.path.exists(_PRISTINE):
                results.append({"argv": ["pristine"], "returncode": 0,
                                "stdout": "cached pristine bundle present", "stderr": "",
                                "elapsed_seconds": 0.0})
                continue
            r = _run(["-m", "harness.abl", "collect", "models/lfm3-1.7b-r8.yaml",
                      "--transcript"], env)
            VOL.commit()
        elif stage == "dir":
            r = _run([f"{_CAMP}/desktop/reproduce_heretic.py", "dir",
                      "--n-each", "400", "--out-dir", "/out/r8"], env)
        elif stage == "edit":
            r = _run([f"{_CAMP}/desktop/reproduce_heretic.py", "edit",
                      "--out-dir", "/out/r8", "--patch", "/out/r8-heretic-repro"], env)
        elif stage == "gate":
            r = _run(["-m", "harness.abl", "collect", "models/lfm3-1.7b-r8.yaml",
                      "--model-dir", "/out/r8-heretic-repro", "--transcript"], env)
        elif stage == "interference":
            r = _run([f"{_CAMP}/desktop/interference_check.py",
                      "--base", "LiquidAI/LFM3-1.7B",
                      "--other", "MihaiPopa-1/LFM3-1.7B-Heretic-Max",
                      "--other", "/out/r8-heretic-repro",
                      "--n", "10", "--out", "/out/r8/interference-honest.json"], env)
        else:
            r = {"argv": [stage], "returncode": 2, "stdout": "", "stderr": "unknown stage"}
        results.append(r)
        VOL.commit()                      # keep artifacts if a later stage dies
        if r["returncode"] != 0:
            results.append({"_stopped_at": stage})
            break
    return {"stages": results, "gpu": GPU}


@app.local_entrypoint()
def main(only: str = ""):
    import json
    stages = [s.strip() for s in only.split(",") if s.strip()] or list(STAGES)
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")
    print(f"running {stages} on {GPU}")
    result = run_rung.remote(stages)
    for r in result["stages"]:
        if "_stopped_at" in r:
            print(f"\n!! stopped at {r['_stopped_at']}")
            continue
        print(f"\n===== {' '.join(r['argv'][-3:])}  rc={r['returncode']}  "
              f"{r['elapsed_seconds']}s")
        print(r["stdout"][-6000:])
        if r["returncode"] != 0:
            print("--- stderr ---")
            print(r["stderr"][-4000:])
    print("\nartifacts: modal volume get absolver-phase2 r8 /tmp/r8")
