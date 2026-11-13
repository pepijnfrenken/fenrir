#!/usr/bin/env bash
# LFM3-1.7B — rung 8: reproduce a published abliteration from disclosed
# parameters and put it through OUR gates. Different strategy = stop searching
# a global alpha over our own direction, and test the one published recipe whose
# numbers are far better than anything six rungs produced.
#
# Recipe provenance (heretic 2.1.0 source + the build's own card):
#   direction: mean(bad) - mean(good) at the FIRST GENERATED TOKEN, per layer,
#              normalised and orthogonalised against the mean(good) direction,
#              then one global direction = lerp(L11, L12, 0.53)
#   edit:      dW = -lambda v (v^T W), row magnitudes restored, lambda from a
#              tent profile -- o_proj 1.19..1.49 peak@14.92, w2 1.09..1.23
#              peak@17.78, layers 11..23, o_proj + w2 only (no conv_out)
#
# Declared falsifiers, written before the numbers exist:
#   F1 direction source: if cos(heretic global, our paired per-layer) < 0.5 the
#      two harvests are not the same object and the campaign's six rungs were
#      measuring a different axis. If > 0.8, direction is not the difference.
#   F2 transfer: if the edited checkpoint reads first-token KL <= 0.05 nats on
#      our meter AND <= 2/12 refusals with the judge calling most responses
#      ANSWER, the recipe transfers and the campaign's NEGATIVE is superseded.
#   F3 published KL: if our meter reads KL >= 0.5 at the disclosed lambda, the
#      published KL 0.0040 is not reproducible under this protocol and the
#      published number cannot be used as a target.
#
# Disk: checkpoint ~3.4 GB under the campaign dir. Nothing deleted; base untouched.
set -uo pipefail
REPO="$HOME/absolver"; D="$REPO/campaigns/liquidai-lfm3-1.7b"; L="$D/desktop/logs"
PY="$REPO/.venv/bin/python"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5; mkdir -p "$L"

exec 9>"$D/.runlock"
if ! flock -n 9; then echo "ABORT: lock held"; exit 6; fi
echo "lock acquired pid $$"

done_() { [ -f "$L/$1.done" ]; }
mark()  { date -u +%FT%TZ > "$L/$1.done"; echo "== phase $1 complete"; }
phase() { local name="$1"; shift
  if done_ "$name"; then echo "== $name: done, skip"; return 0; fi
  echo "== $name: $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee -a "$L/$name.log"; local rc=${PIPESTATUS[0]}
  echo "== $name rc=$rc"; [ "$rc" -eq 0 ] && mark "$name"; return "$rc"; }

p_dir() {
  timeout 5400 "$PY" "$D/desktop/reproduce_heretic.py" dir --n-each 400
}
p_edit() {
  timeout 3600 "$PY" "$D/desktop/reproduce_heretic.py" edit
}
p_gate() {
  timeout 7200 "$PY" harness/abl.py collect models/lfm3-1.7b.yaml \
      --model-dir "$D/r8-heretic-repro" --transcript
}
p_judge() {
  local t
  t=$(find "$D" -name transcript.json -newer "$D/desktop/logs/edit.done" 2>/dev/null | head -1)
  [ -z "$t" ] && { echo "no transcript found"; return 3; }
  echo "judging $t"
  timeout 1800 "$PY" "$D/desktop/judge_transcript.py" "$t" --out "$D/judge-r8-heretic.jsonl"
}

for step in p_dir p_edit p_gate p_judge; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG8 COMPLETE"
