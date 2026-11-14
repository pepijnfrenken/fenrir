"""Rung 22 on Modal: dose sweep on the baked answer-position axis (s = 1.0, 1.25).

Rung 21 shipped the first real artifact: `r21/baked-last-1.5`, no hooks registered, 0 REFUSE /
9 ANSWER / 2 GARBLED on the 12 held-out refusal prompts. It failed R2 on interference only:
benign KL(first answer) mean 0.514 nats against a 0.10 budget. The judged coherence of harmless
responses was intact (0/11/1), so the edit moved the model rather than breaking it.

R3 says the dose is the variable. The rung-19 hook arm at s = 1.0 also cleared the bar
(1 REFUSE / 9 ANSWER / 2 GARBLED), so a smaller dose is a candidate that should cost less KL.

This rung bakes the same axis at s = 1.0 and s = 1.25 and reads both numbers for each:
  refusal pool (12 held-out prompts, no hooks) and benign KL + judged coherence.

Pre-registered:
  D1  a dose that keeps ANSWER >= 8 with GARBLED <= 2 AND benign KL <= 0.10 -> the campaign has a
      usable artifact and it ships at that dose.
  D2  ANSWER falls below 8 exactly where KL comes under budget -> the trade is structural for this
      axis: refusal removal and harmless-prompt drift move together, and the axis is closed at the
      rank-1 uniform form. That gets written down as the campaign's answer, not worked around.
  D3  KL is non-monotonic in dose -> the KL term is not measuring what it claims, and it becomes an
      instrument defect before any further dose is quoted.


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
DOSES = [1.0, 1.25]

if modal.is_local():
    _PROJECT = Path(__file__).resolve().parents[2]
else:
    _PROJECT = Path("/absolver")

app = modal.App("fenrir-lfm3-r22")

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
    _os.makedirs("/out/r22", exist_ok=True)
    print(f"container up; doses={DOSES} dirs={DIRS}", flush=True)
    results = {}
    for dose in DOSES:
        tag = f"{dose:g}"
        opt = f"/out/r22/opt-last-{tag}.json"
        baked = f"/out/r22/baked-last-{tag}"
        d = {}
        d["makeopt"] = _run("makeopt", ["--dirs", DIRS, "--position", "last",
                                        "--strength", str(dose), "--out", opt], env)
        VOL.commit()
        if d["makeopt"]["rc"] != 0:
            results[tag] = d
            continue
        d["bake"] = _run("bake", ["--dirs", DIRS, "--position", "last", "--opt", opt,
                                  "--out-dir", baked], env)
        VOL.commit()
        if d["bake"]["rc"] != 0:
            results[tag] = d
            continue
        d["verify"] = _run("verify", ["--dirs", DIRS, "--position", "last", "--opt", opt,
                                      "--out-dir", baked, "--max-new", "4096",
                                      "--out", f"/out/r22/verify-last-{tag}.json"], env)
        VOL.commit()
        d["benign"] = _run("benign", ["--dirs", DIRS, "--position", "last", "--opt", opt,
                                      "--out-dir", baked, "--max-new", "512",
                                      "--n-harmless", "12",
                                      "--out", f"/out/r22/benign-last-{tag}.json"], env)
        VOL.commit()
        results[tag] = d
    return results


@app.local_entrypoint()
def main():
    res = run_rung.remote()
    print(json.dumps(res, indent=1))
