#!/usr/bin/env bash
# LFM3-1.7B — rung 4: is the direction a refusal axis or a thinking-vs-answering axis?
#
# Evidence so far: refusal falls with edit magnitude (11/9/7 of 12 at alpha
# 0.5/1.0/2.0) but first-token KL is 0.65/0.99/1.39 against a 0.1 cap, i.e. the
# "removal" is proportional to the damage. Attention-only at alpha 2.0 drops KL
# to 0.326 and removes nothing (12/12). Steering the direction at +/-1 collapses
# the decoder at both signs.
#
# Hypothesis: the paired harvest used paired_max_new_tokens=96 while LFM3
# reasons before answering, so the unprimed half was sampled mid-thought and the
# direction encodes "still thinking" vs "answer now", not refuse vs comply.
# Test: re-harvest with a 320-token budget and check sign selectivity. A real
# refusal axis lowers refusal at -alpha and raises it at +alpha without collapse.
# No weight edits in this rung.
set -uo pipefail
REPO="$HOME/absolver"; D="$REPO/campaigns/liquidai-lfm3-1.7b"; L="$D/desktop/logs"
CFG="models/lfm3-1.7b.yaml"; PY="$REPO/.venv/bin/python"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5; mkdir -p "$L"
done_() { [ -f "$L/$1.done" ]; }
mark()  { date -u +%FT%TZ > "$L/$1.done"; echo "== phase $1 complete"; }
phase() { local name="$1"; shift
  if done_ "$name"; then echo "== $name: done, skip"; return 0; fi
  echo "== $name: $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee -a "$L/$name.log"; local rc=${PIPESTATUS[0]}
  echo "== $name rc=$rc"; [ "$rc" -eq 0 ] && mark "$name"; return "$rc"; }

p_keep_old() {
  [ -f "$D/directions-chat-paired-t96.pt" ] || cp "$D/directions-chat-paired.pt" "$D/directions-chat-paired-t96.pt"
  ls -l "$D"/directions-chat-paired*.pt
}
p_harvest320() { timeout 5400 "$PY" harness/abl.py directions "$CFG" --dir-method paired; }
p_rename() {
  mv -f "$D/directions-chat-paired.pt" "$D/directions-chat-paired-t320.pt"
  ls -l "$D"/directions-chat-paired*.pt
}
p_steer320() {
  timeout 5400 "$PY" harness/abl.py steer-test "$CFG" \
      --from-directions "$D/directions-chat-paired-t320.pt" \
      --alphas=-2,-1,0,1,2 --n-directions 10 --n-prompts 8 --n-benign 6
}

for step in p_keep_old p_harvest320 p_rename p_steer320; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG4 COMPLETE"
