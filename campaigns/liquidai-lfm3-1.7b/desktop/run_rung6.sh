#!/usr/bin/env bash
# LFM3-1.7B — rung 6: does the direction carry refusal at the REAL operating
# scale, and was rung 5's "garbled" an artefact of a 256-token window?
#
# What rung 5 established, none of it predicted beforehand:
#   1. The model's operating scale is ~0.05, not 0.5-2. At alpha 0.2 and above
#      ANY direction destroys it, including `ctrl-random`, a gaussian vector
#      with no relation to refusal (0/8 coherent at -1, 0.5, 1, 2).
#      => rung 4's argument ("collapses at both signs, therefore not a refusal
#      axis") is RETRACTED. It was measuring the magnitude, not the direction.
#   2. At alpha 0.05 and 0.10 the refusal sentence disappears and coherence
#      (the old detector) reads 8/8. A blind LLM judge labels 16/16 of those
#      responses GARBLED: shape of an answer, no answer.
#   3. The judge also labelled 4/6 benign baseline responses GARBLED, because at
#      256 tokens the model is still inside <think>. So the gate window was
#      scoring the preamble. Config now uses 1024.
#
# Rung 6 is the specificity test at the operating scale, at a real window:
#   A. real direction at -0.1, -0.05, 0.05, 0.1 (sign selectivity at the scale
#      where the model does not simply die)
#   B. ctrl-random and ctrl-shuf at 0.05 and 0.1 (semantically empty directions,
#      same magnitude)
# Declared falsifier, written before the run:
#   If ctrl-random at 0.05 also removes the refusal sentence, the direction
#   carries no refusal information and the campaign's remaining positive lead
#   dies. If real 0.05 removes refusal and random 0.05 does not, the direction
#   is specific and the weight edit deserves a matched small alpha.
# No weight edits in this rung. Single-writer lock.
set -uo pipefail
REPO="$HOME/absolver"; D="$REPO/campaigns/liquidai-lfm3-1.7b"; L="$D/desktop/logs"
CFG="models/lfm3-1.7b.yaml"; PY="$REPO/.venv/bin/python"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5; mkdir -p "$L"

exec 9>"$D/.runlock"
if ! flock -n 9; then
  echo "ABORT: another runner holds $D/.runlock"; exit 6
fi
echo "lock acquired: $D/.runlock pid $$"

done_() { [ -f "$L/$1.done" ]; }
mark()  { date -u +%FT%TZ > "$L/$1.done"; echo "== phase $1 complete"; }
phase() { local name="$1"; shift
  if done_ "$name"; then echo "== $name: done, skip"; return 0; fi
  echo "== $name: $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee -a "$L/$name.log"; local rc=${PIPESTATUS[0]}
  echo "== $name rc=$rc"; [ "$rc" -eq 0 ] && mark "$name"; return "$rc"; }

GEN=1024
p_sign1024() {
  timeout 10800 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/directions-chat-paired-t320.pt" \
      --alphas=-0.1,-0.05,0.05,0.1 --max-new-tokens $GEN \
      --n-directions 10 --n-prompts 8 --n-benign 6
}
p_ctrl_small() {
  timeout 10800 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/dirs-ctrl-random.pt" \
      --alphas=0.05,0.1 --max-new-tokens $GEN \
      --n-directions 10 --n-prompts 8 --n-benign 6
}
p_shuf_small() {
  timeout 10800 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/dirs-ctrl-shuf.pt" \
      --alphas=0.05,0.1 --max-new-tokens $GEN \
      --n-directions 10 --n-prompts 8 --n-benign 6
}

for step in p_sign1024 p_ctrl_small p_shuf_small; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG6 COMPLETE"
