#!/usr/bin/env bash
# LFM3-1.7B — rung 3: like-for-like pristine baseline + one mechanism test.
#
# Rung 2 verdict: refusal falls monotonically with alpha (11/12, 9/12, 7/12 at
# alpha 0.5/1.0/2.0) but first-token KL is 0.65/0.99/1.39 against a 0.1 cap, so
# no alpha satisfies both. Empty operating window.
#
# Mechanism question worth one run: the edit touched 13 conv layers, 6 full
# attention layers and 5 sliding-attention layers, all with the same direction.
# If the KL damage comes from writing the direction into the conv stack (whose
# out_proj is followed by a post-norm, i.e. a different residual path), then an
# attention-only edit at the same alpha should keep refusal behaviour while
# holding KL down. One config, declared before the run.
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

# 1. pristine gates on the SAME 12-prompt split (overwrites the 5-prompt baseline)
p_pristine12() { timeout 7200 "$PY" harness/abl.py collect "$CFG" --transcript; }

# 2. attention-only edit: the 6 full-attention layers (3,7,11,15,19,23) and w2
p_abl_attn() {
  timeout 3600 "$PY" harness/abl.py abl "$CFG" \
      --method mpoa --alpha 2.0 --layers 3,7,11,15,19,23 \
      --weights o_proj,w2 --from-directions "$DIRS" --tag mpoa-a2.0-attnonly
}
p_collect_attn() {
  timeout 7200 "$PY" harness/abl.py collect "$CFG" \
      --model-dir "$D/mpoa-a2.0-attnonly" --transcript
}

for step in p_pristine12 p_abl_attn p_collect_attn; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG3 COMPLETE"
