#!/usr/bin/env bash
# LFM3-1.7B — rung 2: bigger gates + alpha ladder.
#
# Why: the first pass ran a 5-prompt held-out gate (hardcoded n_test=5) and a
# 3-prompt steer test. That resolves 20pp and 33pp steps — too coarse to call
# anything. This rung reruns the SAME direction and edit with a 12-prompt
# held-out gate, then walks the alpha ladder down because alpha 2.0 blew the
# first-token KL gate (1.68 nats) while refusal stayed at 4/5.
#
# Runs inside the desktop WSL distro. Skip-if-done per phase.
set -uo pipefail
REPO="$HOME/absolver"
D="$REPO/campaigns/liquidai-lfm3-1.7b"
L="$D/desktop/logs"
CFG="models/lfm3-1.7b.yaml"
PY="$REPO/.venv/bin/python"
DIRS="$D/directions-chat-paired.pt"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5
mkdir -p "$L"

done_() { [ -f "$L/$1.done" ]; }
mark()  { date -u +%FT%TZ > "$L/$1.done"; echo "== phase $1 complete"; }
phase() {
  local name="$1"; shift
  if done_ "$name"; then echo "== $name: done, skip"; return 0; fi
  echo "== $name: $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee -a "$L/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "== $name rc=$rc"
  [ "$rc" -eq 0 ] && mark "$name"
  return "$rc"
}

# 1. steer gate at 12 harmful + 6 benign, 5 alphas (deduped 0.0)
p_steer12() {
  timeout 7200 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$DIRS" --alphas=-2,-1,0,1,2 \
      --n-directions 10 --n-prompts 12 --n-benign 6 --require-effect
}

# 2. re-collect the existing alpha-2.0 model under the 12-prompt split
p_collect_a20() {
  timeout 7200 "$PY" harness/abl.py collect "$CFG" \
      --model-dir "$D/mpoa-a2.0-all24" --transcript
}

# 3/4. build + collect alpha 1.0
p_abl_a10() {
  timeout 3600 "$PY" harness/abl.py abl "$CFG" \
      --method mpoa --alpha 1.0 --layers 0-23 \
      --weights o_proj,conv_out,w2 --from-directions "$DIRS" --tag mpoa-a1.0-all24
}
p_collect_a10() {
  timeout 7200 "$PY" harness/abl.py collect "$CFG" \
      --model-dir "$D/mpoa-a1.0-all24" --transcript
}

# 5/6. build + collect alpha 0.5
p_abl_a05() {
  timeout 3600 "$PY" harness/abl.py abl "$CFG" \
      --method mpoa --alpha 0.5 --layers 0-23 \
      --weights o_proj,conv_out,w2 --from-directions "$DIRS" --tag mpoa-a0.5-all24
}
p_collect_a05() {
  timeout 7200 "$PY" harness/abl.py collect "$CFG" \
      --model-dir "$D/mpoa-a0.5-all24" --transcript
}

for step in p_steer12 p_collect_a20 p_abl_a10 p_collect_a10 p_abl_a05 p_collect_a05; do
  phase "${step#p_}" "$step"; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!! stop at $step rc=$rc"
    # steer returning 3 = direction not causal at 12 prompts: that is a result,
    # not a crash. Keep going to the collects so the gate table still exists.
    if [ "$step" = "p_steer12" ] && [ "$rc" -eq 3 ]; then
      echo "!! steer gate says NOT CAUSAL at 12 prompts; continuing to collects (recorded as such)"
      continue
    fi
    exit "$rc"
  fi
done

echo "== LADDER COMPLETE"
"$PY" - "$D" <<'EOF'
import json, glob
print(f"{'target':22} {'refusal':>9} {'coh':>5} {'deg':>5} {'cap':>6} {'ppl':>8} {'kl':>7} pass")
for f in sorted(glob.glob('/home/pino/absolver/campaigns/liquidai-lfm3-1.7b/collect-*/bundle.json')):
    j = json.load(open(f)); g = {k: (j.get(k) or {}) for k in
        ("refusal","coherence","degeneracy","capability","perplexity_increase","first_token_kl")}
    print(f"{f.split('/')[-2][:22]:22} {str(g['refusal'].get('value')):>9} "
          f"{str(g['coherence'].get('value')):>5} {str(g['degeneracy'].get('value')):>5} "
          f"{str(g['capability'].get('value')):>6} "
          f"{str(g['perplexity_increase'].get('value'))[:7]:>8} "
          f"{str(g['first_token_kl'].get('value'))[:6]:>7} {j['eval_pass']}")
EOF
