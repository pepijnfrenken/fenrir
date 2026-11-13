#!/usr/bin/env bash
# LFM3-1.7B abliteration campaign — phased, skip-if-done, evidence-gated.
# Runs INSIDE the desktop WSL distro (Ubuntu-24.04, user pino).
# Transport: ssh win 'wsl -d Ubuntu-24.04 -u pino -- bash -s' < this_file
#
# Phase order follows the repo contract: instrument sanity on the PRISTINE model
# before any weight edit, then the causal steer gate, then ONE config, then gates.
# Exit codes: 2 = baseline sanity red (instrument suspect), 3 = direction not
# causal, 4 = gate failure, 5 = setup problem. 0 = all phases passed.
set -uo pipefail

REPO="$HOME/absolver"
D="$REPO/campaigns/liquidai-lfm3-1.7b"
L="$D/desktop/logs"
CFG="models/lfm3-1.7b.yaml"
PY="$REPO/.venv/bin/python"
DIRS="$D/directions-chat-paired.pt"
TAG="mpoa-a2.0-all24"
MODELDIR="$D/$TAG"

mkdir -p "$L"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/hf"
export TOKENIZERS_PARALLELISM=false
cd "$REPO" || { echo "FATAL: $REPO missing"; exit 5; }

done_() { [ -f "$L/$1.done" ]; }
mark()  { date -u +%FT%TZ > "$L/$1.done"; echo "== phase $1 complete"; }

phase() {
  local name="$1"; shift
  if done_ "$name"; then echo "== $name: already done, skipping"; return 0; fi
  echo "== $name: $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee -a "$L/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "== $name rc=$rc"
  [ "$rc" -eq 0 ] && mark "$name"
  return "$rc"
}

# ---------------------------------------------------------------- 0. environment
p0_setup() {
  python3 -m venv .venv || return 5
  .venv/bin/pip install -q --upgrade pip
  # driver 561.17 = CUDA 12.6 on this box; cu126 wheels match it.
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126 || return 5
  .venv/bin/pip install -q transformers accelerate datasets safetensors \
      pydantic pyyaml huggingface-hub numpy requests || return 5
  "$PY" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("transformers", transformers.__version__)
EOF
}

# ------------------------------------------------------------------- 1. inspect
p1_inspect() { timeout 2400 "$PY" harness/abl.py inspect "$CFG"; }

# ----------------------------------------------------- 2. directions (paired)
p2_directions() { timeout 3600 "$PY" harness/abl.py directions "$CFG" --dir-method paired; }

# ------------------------------------- 3. pristine collect: baseline sanity first
p3_baseline() {
  timeout 3600 "$PY" harness/abl.py collect "$CFG" --transcript
  local rc=$?
  [ "$rc" -eq 2 ] && echo "BASELINE SANITY RED — instrument suspect, no ablation is validatable" >&2
  return "$rc"
}

# ------------------------------------------------- 4. causal steer gate (exit 3)
p4_steer() {
  timeout 3600 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$DIRS" \
      --alphas=-3,-2,-1,0,1,2,3 --n-directions 10 --n-prompts 3 \
      --require-effect
}

# --------------------------------------------- 5. ONE weight edit (mpoa, all 24)
p5_abl() {
  timeout 3600 "$PY" harness/abl.py abl "$CFG" \
      --method mpoa --alpha 2.0 --layers 0-23 \
      --weights o_proj,conv_out,w2 \
      --from-directions "$DIRS" --tag "$TAG"
}

# ------------------------------------------------------- 6. gates on the edit
p6_collect() { timeout 3600 "$PY" harness/abl.py collect "$CFG" --model-dir "$MODELDIR" --transcript; }

for step in p0_setup p1_inspect p2_directions p3_baseline p4_steer p5_abl p6_collect; do
  phase "${step#p?_}" "$step"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!! stopping at $step (rc=$rc)"
    exit "$rc"
  fi
done

echo "== CAMPAIGN RUN COMPLETE"
"$PY" - "$D" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for b in sorted(d.glob("collect-*/bundle.json")):
    j = json.loads(b.read_text())
    g = j.get("gate_report", {}) or {}
    print(f"\n== {b.parent.name} ==")
    print("  eval_pass:", g.get("eval_pass"))
    for k in ("refusal", "coherence", "degeneracy", "capability",
              "ppl_increase", "first_token_kl", "baseline_sanity"):
        if k in g:
            print(f"  {k}: {json.dumps(g[k])[:300]}")
    if "refusal" in j:
        print("  refusal block:", json.dumps(j["refusal"])[:400])
EOF
