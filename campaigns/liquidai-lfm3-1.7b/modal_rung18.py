"""Rung 18 on Modal: optimise the per-place strengths, gate them, bake them, verify the bake.

Four stages in one L4 container, in this order, each streamed:

  optim   48 strengths fitted on 20 harmful / 20 harmless prompts the gate does not use
  gate    the same strengths applied at generation time on the 12 held-out prompts
  bake    W' = W - s a (a^T W) written into a real checkpoint under /out/r18/baked
  verify  the baked checkpoint loaded with NO hooks, same 12 prompts

Why all four. Rungs 12-14b closed the fixed-ratio family: one strength for every place walks
a single refusal-for-garble curve. The public builds that clear the bar are per-place
optimised against a KL budget. `gate` tests the operator; `verify` tests the artifact. A model
only exists after `verify`.

Pre-registered before the numbers exist:
  B1  the operator works iff gate shows judge ANSWER >= 8/12 with GARBLED <= 2 and
      REFUSE <= 2. Anything else and the optimiser form fails like the fixed-ratio form.
  B2  the artifact is real iff the baked arm's ANSWER count is within 2 of the gate arm's,
      i.e. the rank-1 pre-norm bake reproduces the hooked operator. This is the falsifier
      rung 9 failed at the post-norm endpoint (F1 0.998, 0.99997); at the pre-norm endpoint
      the map is exact, so a large gap here means the derivation or the tensor map is wrong.
  B3  failure of B1 with a flat dose curve at the KL budget closes the campaign's own
      recipe at every form: fixed-ratio, prompt-only, dose-swept, and optimised.

Cost: one L4, four model loads, budget ceiling 90 minutes.
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

app = modal.App("fenrir-lfm3-r18")

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
            "tail": "".join(lines)[-4000:]}


@app.function(image=image, gpu=GPU, timeout=10800, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOL})
def run_rung(steps: int = 200, lam: float = 3.0, lr: float = 0.05) -> dict:
    import json
    import os as _os
    import time

    env = dict(_os.environ)
    env.update({"PYTHONPATH": "/absolver", "ABSOLVER_CAMPAIGNS_ROOT": "/out",
                "HF_HOME": "/out/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    _os.makedirs("/out/r18", exist_ok=True)
    print(f"container up; steps={steps} lam={lam} lr={lr}", flush=True)
    t0 = time.perf_counter()
    results = {}

    r = _run("optim", ["--steps", str(steps), "--lam", str(lam), "--lr", str(lr),
                       "--out", "/out/r18/opt.json"], env)
    results["optim"] = r
    VOL.commit()
    if r["rc"] != 0:
        print(f"FAIL FAST: optim rc={r['rc']}")
        return {"stopped": "optim", "results": results}

    opt = json.loads(open("/out/r18/opt.json").read())
    wv = [float(v) for v in opt["w"].values()]
    moved = sum(1 for v in wv if v > 1e-3)
    print(f"optim moved {moved}/{len(wv)} places; w max {max(wv):.3f}")
    if moved == 0:
        print("FAIL FAST: optimiser left every strength at zero — nothing to gate")
        return {"stopped": "empty_optim", "results": results}

    r = _run("gate", ["--opt", "/out/r18/opt.json", "--max-new", "4096",
                      "--out", "/out/r18/gate.json"], env)
    results["gate"] = r
    VOL.commit()
    if r["rc"] != 0:
        print(f"gate rc={r['rc']} — stopping before the bake")
        return {"stopped": "gate", "results": results}

    r = _run("bake", ["--opt", "/out/r18/opt.json", "--out-dir", "/out/r18/baked"], env)
    results["bake"] = r
    VOL.commit()
    if r["rc"] != 0:
        print(f"bake rc={r['rc']} — no artifact")
        return {"stopped": "bake", "results": results}

    r = _run("verify", ["--out-dir", "/out/r18/baked", "--max-new", "4096",
                        "--out", "/out/r18/verified.json"], env)
    results["verify"] = r
    VOL.commit()
    print(f"total {round(time.perf_counter() - t0, 1)}s", flush=True)
    return {"finished": True, "results": results}


@app.local_entrypoint()
def main(steps: int = 200, lam: float = 3.0, lr: float = 0.05):
    out = run_rung.remote(steps, lam, lr)
    if out.get("stopped"):
        print(f"STOPPED at {out['stopped']}")
    for k, v in (out.get("results") or {}).items():
        print(f"{k}: rc={v['rc']} in {v['elapsed']}s")
