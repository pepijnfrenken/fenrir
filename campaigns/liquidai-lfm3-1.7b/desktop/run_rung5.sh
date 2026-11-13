#!/usr/bin/env bash
# LFM3-1.7B — rung 5: null-direction controls + the low-alpha floor.
#
# Two gaps in the campaign so far, both declared BEFORE this run:
#
# A. LOW-ALPHA FLOOR. Every alpha tested was >= 0.5. The KL cap is 0.1 and the
#    smallest alpha already overshot it 6.5x, so "no operating window" was only
#    demonstrated from above. If refusal survives at 0.05..0.35 while coherence
#    holds, the window is empty from below too and the verdict is sealed.
#
# B. NULL-DIRECTION CONTROL. Rung 4 read "collapses at both signs" as "not a
#    refusal axis". That inference needs a control: does a semantically empty
#    direction of the same magnitude also destroy the decoder? sweep.py
#    normalises d and applies -d*alpha, so alpha is absolute hidden-state
#    units and a unit random vector is exactly comparable.
#      - ctrl-random: per-layer gaussian unit vector (seed 20261113)
#      - ctrl-shuf  : real vectors, layer assignment deranged (seed 20261113)
#    If BOTH controls collapse at the same alphas as the real direction, then
#    "not sign-selective" says nothing about refusal and must be retracted.
#    If only the real direction collapses, the NEGATIVE verdict strengthens.
#
# No weight edits in this rung. One lock: two drivers on one 8GB card already
# raced once (proc_87c410692949 was SIGTERMed mid-harvest).
set -uo pipefail
REPO="$HOME/absolver"; D="$REPO/campaigns/liquidai-lfm3-1.7b"; L="$D/desktop/logs"
CFG="models/lfm3-1.7b.yaml"; PY="$REPO/.venv/bin/python"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5; mkdir -p "$L"

# ---- single-writer lock -------------------------------------------------
exec 9>"$D/.runlock"
if ! flock -n 9; then
  echo "ABORT: another runner holds $D/.runlock (pid $(cat "$D/.runlock" 2>/dev/null))"
  exit 6
fi
echo "lock acquired: $D/.runlock pid $$"

done_() { [ -f "$L/$1.done" ]; }
mark()  { date -u +%FT%TZ > "$L/$1.done"; echo "== phase $1 complete"; }
phase() { local name="$1"; shift
  if done_ "$name"; then echo "== $name: done, skip"; return 0; fi
  echo "== $name: $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee -a "$L/$name.log"; local rc=${PIPESTATUS[0]}
  echo "== $name rc=$rc"; [ "$rc" -eq 0 ] && mark "$name"; return "$rc"; }

p_ctrl_build() { timeout 900 "$PY" "$D/desktop/make_controls.py"; }

# Test A: real direction, small magnitudes. Same 8 harmful / 6 benign split as
# rung 4 so the baseline row is comparable.
p_lowalpha() {
  timeout 7200 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/directions-chat-paired-t320.pt" \
      --alphas=0.05,0.1,0.2,0.35,0.5 --n-directions 10 --n-prompts 8 --n-benign 6
}

# Test B: controls. The alphas bracket the real direction's whole ladder, plus
# 0.5 where the real direction still produced coherent text (11/12 refused).
p_ctrl_random() {
  timeout 7200 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/dirs-ctrl-random.pt" \
      --alphas=-1,0.5,1,2 --n-directions 10 --n-prompts 8 --n-benign 6
}
p_ctrl_shuf() {
  timeout 7200 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/dirs-ctrl-shuf.pt" \
      --alphas=-1,0.5,1,2 --n-directions 10 --n-prompts 8 --n-benign 6
}

for step in p_ctrl_build p_lowalpha p_ctrl_random p_ctrl_shuf; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG5 COMPLETE"
