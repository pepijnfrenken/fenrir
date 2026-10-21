"""Modal guard-4b lane: pristine acceptance + probe + steer (runbook steps 1–3).

Scale-up runbook item: guard_visibility_price --n 40 --reuse-gen on the PRISTINE
Qwen3Guard-Gen-4B, then run_guard_gates, via campaign scripts — no repo changes.
"""
import modal
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11", force_build=True)
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30", "pydantic>=2", "pyyaml>=6",
        "huggingface-hub>=0.20", "numpy>=1.24", "accelerate>=0.20",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(
        str(PROJECT_DIR), remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                ".pytest_cache", ".mypy_cache", ".ruff_cache",
                "abliterated_models", "experiments", "build"],
    )
    .add_local_file(str(PROJECT_DIR / "models" / "qwen3guard-4b.yaml"),
                    "/absolver/models/qwen3guard-4b.yaml")
)

app = modal.App("fenrir-guard-4b")

VOL = modal.Volume.from_name("fenrir-guard-4b", create_if_missing=True)

@app.function(
    image=image, gpu="A100", timeout=7200, retries=0,
    secrets=[modal.Secret.from_name("hf-write-token")],
    volumes={"/work": VOL},
)
def guard_lane(stage: str = "pristine-gates") -> dict:
    os.chdir("/absolver")
    import subprocess, sys, json
    env = dict(os.environ)
    os.makedirs("/work/results", exist_ok=True)
    gl = campaigns_dir = "/absolver/campaigns/guard-lane"
    if stage == "pristine-gates":
        cmd = [sys.executable, f"{gl}/guard_visibility_price.py",
               "--guard", "Qwen/Qwen3Guard-Gen-4B",
               "--n", "40", "--reuse-gen"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return {"stage": stage, "rc": r.returncode,
                "stdout_tail": r.stdout[-2500:], "stderr_tail": r.stderr[-2500:]}
    elif stage == "smoke":
        cmd = [sys.executable, f"{gl}/guard_visibility_price.py",
               "--guard", "Qwen/Qwen3Guard-Gen-4B", "--smoke"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return {"stage": stage, "rc": r.returncode,
                "stdout_tail": r.stdout[-2500:], "stderr_tail": r.stderr[-2500:]}
    elif stage == "acceptance":
        cmd = [sys.executable, f"{gl}/run_guard_gates.py",
               "--pairs", f"{gl}/data/guard_pairs_lfm-abl_slice2_n120.jsonl",
               "--split", "test", "--config", "models/qwen3guard-4b.yaml",
               "--out", "/work/results/guard_gates_4b_test.json"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return {"stage": stage, "rc": r.returncode,
                "stdout_tail": r.stdout[-2500:], "stderr_tail": r.stderr[-2500:]}
    elif stage == "repl":
        # runbook step 5: fresh-slice replication on the SAVED ablated checkpoint.
        # Visibility price runs on the checkpoint as the guard (the ablation target
        # is the guard itself, not the subject). Writes into /work, mirrored back.
        volstaging = "/work/results"
        os.system(f"mkdir -p {volstaging}")
        # stage the ablated checkpoint into the image-local dir expected by the paired-path tooling
        if not os.path.isdir("/absolver/abliterated_models/guard4b-abl-l32-a6.0"):
            os.system("mkdir -p /absolver/abliterated_models && "
                      "cp -r /work/checkpoints/guard4b-abl-l32-a6.0 /absolver/abliterated_models/")
        cmd = [sys.executable, f"{gl}/guard_visibility_price.py",
               "--guard", "/absolver/abliterated_models/guard4b-abl-l32-a6.0",
               "--n", "40", "--offset", "40",
               "--subjects", "pristine,ablated"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/absolver")
        probedir = "/absolver/campaigns/guard-lane/results"
        if os.path.isdir(probedir):
            for fn in os.listdir(probedir):
                if "off40" in fn or "lfm_pair_n40x2" in fn:
                    os.system(f"cp {probedir}/{fn} {volstaging}/cp-{fn} 2>/dev/null || "
                              f"cp {probedir}/{fn} {volstaging}/")
        return {"stage": "repl", "rc": r.returncode,
                "stdout_tail": r.stdout[-3500:], "stderr_tail": r.stderr[-1500:]}
    elif stage == "probe":
        # runbook step 3: harvest → distill on the 4B's own layers.
        # note: distill has no --config; it re-reads the npz written by harvest
        # under --tag, so only harvest needs the 4B config.
        for sub, extra in (("harvest", ["--config", "models/qwen3guard-4b.yaml", "--tag", "guard4b"]),
                           ("distill", ["--tag", "guard4b"])):
            cmd = [sys.executable, f"{gl}/guard_probe.py", sub] + extra
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/absolver")
            if r.returncode != 0:
                return {"stage": sub, "rc": r.returncode,
                        "stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]}
            # persist probe artifacts to the durable volume (RESULTS is image-local)
            probedir = "/absolver/campaigns/guard-lane/results"
            if os.path.isdir(probedir):
                for fn in os.listdir(probedir):
                    if "guard4b" in fn:
                        os.system(f"cp {probedir}/{fn} /work/results/")
        return {"stage": "harvest+distill", "rc": 0,
                "stdout_tail": "both subcommands rc=0"}
    elif stage.startswith("excise-save"):
        # checkpoint stage: single group at one alpha, WITH save to the durable volume.
        # format: excise-save:<layers>|<alpha>  e.g. excise-save:32|6.0
        volstaging = "/work/results"
        probedir = "/absolver/campaigns/guard-lane/results"
        os.makedirs(probedir, exist_ok=True)
        for fn in ("probe_guard4b.npz", "directions_guard4b.npz",
                   "directions_guard4b_profile.json"):
            src = os.path.join(volstaging, fn)
            if os.path.exists(src):
                os.system(f"cp {src} {probedir}/{fn}")
        spec = stage.split(":", 1)[1]
        layers_g, alpha_g = spec.split("|")
        out_dir = f"/work/checkpoints/guard4b-abl-l{layers_g.replace(',','-')}-a{alpha_g}"
        cmd = [sys.executable, f"{gl}/guard_excise.py",
               "--config", "models/qwen3guard-4b.yaml",
               "--tag", "guard4b", "--layers", layers_g,
               "--alpha", str(float(alpha_g)),
               "--pairs", f"{gl}/data/guard_pairs_lfm-abl_slice2_n120.jsonl",
               "--out", out_dir]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/absolver")
        for fn in os.listdir(probedir):
            if "excise" in fn and "guard4b" in fn:
                os.system(f"cp {probedir}/{fn} {volstaging}/")
        # Modal volumes commit lazily; force-commits + verify by reading the weight shard back
        import glob as _glob
        shards = _glob.glob(f"{out_dir}/*.safetensors")
        ok = True
        for s in shards:
            with open(s, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")   # safetensors: header len
                fh.seek(n + 8)
                fh.read(1)                                 # touch last byte -> forces fs sync too
            ok = ok and os.path.getsize(s) > 4e9
        return {"stage": stage, "rc": r.returncode, "shards": shards, "shard_ok": ok,
                "stdout_tail": r.stdout[-2500:], "stderr_tail": r.stderr[-800:],
                "out_dir": out_dir}
    elif stage.startswith("excise"):
        # runbook step 4: weight-edit candidate grid. Format excise:L20,L22:L26,L32
        #   ':' separates alpha groups; each group's layers get the same alpha.
        #   Default (no colon): all listed layers at the single grid alpha.
        # steer artifacts are staged from the durable volume like the steer stage.
        volstaging = "/work/results"
        probedir = "/absolver/campaigns/guard-lane/results"
        os.makedirs(probedir, exist_ok=True)
        for fn in ("probe_guard4b.npz", "directions_guard4b.npz",
                   "directions_guard4b_profile.json"):
            src = os.path.join(volstaging, fn)
            if os.path.exists(src):
                os.system(f"cp {src} {probedir}/{fn}")
        spec = stage.split(":", 1)[1] if ":" in stage else ""
        import shlex as _sh
        grid = [g for g in spec.split(";") if g.strip()]
        results = []
        for g in grid:
            parts = [x for x in g.split("|") if x.strip()]
            if len(parts) == 2:
                layers_g, alpha_g = parts[0].strip(), float(parts[1].strip())
            else:
                layers_g, alpha_g = g.strip(), float(os.environ.get("EXCISE_ALPHA", "1.0"))
            cmd = [sys.executable, f"{gl}/guard_excise.py",
                   "--config", "models/qwen3guard-4b.yaml",
                   "--tag", "guard4b", "--layers", layers_g,
                   "--alpha", str(alpha_g),
                   "--pairs", f"{gl}/data/guard_pairs_lfm-abl_slice2_n120.jsonl",
                   "--no-save"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/absolver")
            results.append({"layers": layers_g, "alpha": alpha_g, "rc": r.returncode,
                            "stdout_tail": r.stdout[-1200:], "stderr_tail": r.stderr[-1200:]})
            for fn in os.listdir(probedir):
                if "excise" in fn and "guard4b" in fn:
                    os.system(f"cp {probedir}/{fn} {volstaging}/")
            if r.returncode != 0:
                break  # stop the grid on first failure; per-group results returned
        return {"stage": f"excise:{spec}", "groups": results}
    elif stage.startswith("steer"):
        # runbook step 3b: steer-test candidate layers from the distillation.
        # stage format: steer:L26,L34 (layers). steer re-reads probe_/directions_
        # npz shipped from the DURABLE volume (image-local results dir evaporates).
        layers = stage.split(":", 1)[1] if ":" in stage else "34,33,32"
        volstaging = "/work/results"
        probedir = "/absolver/campaigns/guard-lane/results"
        os.makedirs(probedir, exist_ok=True)
        for fn in ("probe_guard4b.npz", "directions_guard4b.npz",
                   "directions_guard4b_profile.json"):
            src = os.path.join(volstaging, fn)
            if os.path.exists(src):
                os.system(f"cp {src} {probedir}/{fn}")
        for L in layers.split(","):
            cmd = [sys.executable, f"{gl}/guard_probe.py", "steer",
                   "--config", "models/qwen3guard-4b.yaml", "--tag", "guard4b",
                   "--layer", L]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/absolver")
            if r.returncode != 0:
                return {"stage": f"steer:{L}", "rc": r.returncode,
                        "stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]}
            for fn in os.listdir(probedir):
                if "steer" in fn and "guard4b" in fn:
                    os.system(f"cp {probedir}/{fn} {volstaging}/")
        return {"stage": f"steer:{layers}", "rc": 0,
                "stdout_tail": "steer subcommands rc=0"}
    return {"stage": stage, "error": "unknown stage"}

@app.local_entrypoint()
def main(stage: str = "pristine-gates"):
    out = guard_lane.remote(stage)
    import json
    print(json.dumps(out, indent=1)[:3500])
