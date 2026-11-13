#!/usr/bin/env bash
# LFM3-1.7B — rung 7: measure a PUBLIC abliterated build with OUR instruments,
# and recover its edit in weight space. Different strategy = stop guessing a
# global alpha and find out what the builds that report KL 0.004 actually did.
#
# Public disclosure (MihaiPopa-1/LFM3-1.7B-Heretic-Max, Heretic v1.4.0):
#   direction_index 11.53 (fractional -> blend of neighbouring layer directions)
#   attn.o_proj      min 1.19 .. max 1.49, peak at layer 14.92, reach 13.28
#   mlp.down_proj    min 1.09 .. max 1.23, peak at layer 17.78, reach 11.61
#   own report: KL 0.0040, refusals 6/100 (base 93/100)
# That is a per-layer tent profile over o_proj + down_proj w2 only — no conv_out,
# and no single global alpha. Nothing this campaign tried looked like that.
#
# Two questions, in order:
#   A. Does the public build survive OUR gates? style-aware refusal at a
#      1024-token window on the same 12-prompt held-out split, plus coherence,
#      degeneracy and capability. If it reads 12/12 refused, then 6/100 was a
#      keyword measurement and the ecosystem number is not a target.
#   B. What is the edit made of? per-tensor rel_change, rank-1 fraction, effective
#      rank, row-norm drift vs base. A rank-1 projection reads rank1~1.0.
#
# No weight edits here. Single-writer lock. Read-only on the base model.
set -uo pipefail
REPO="$HOME/absolver"; D="$REPO/campaigns/liquidai-lfm3-1.7b"; L="$D/desktop/logs"
PY="$REPO/.venv/bin/python"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5; mkdir -p "$L" "$D/public"

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

# A. gate the public build
p_pub_gate() {
  timeout 7200 "$PY" harness/abl.py collect models/lfm3-1.7b-pub-heretic.yaml --transcript
}

# B. weight-space diff. Base comes from the HF cache the campaign already built.
p_fetch() {
  mkdir -p "$D/public/heretic-max" "$D/public/ara"
  "$PY" - <<'PY'
from huggingface_hub import snapshot_download
import os
for repo, dest in [("MihaiPopa-1/LFM3-1.7B-Heretic-Max", "public/heretic-max"),
                   ("Austriani/LFM3-1.7B-ARA-Abliterated", "public/ara"),
                   ("LiquidAI/LFM3-1.7B", "public/base")]:
    root = os.path.expanduser(f"~/absolver/campaigns/liquidai-lfm3-1.7b/{dest}")
    p = snapshot_download(repo_id=repo, local_dir=root,
                          allow_patterns=["*.safetensors", "*.json", "*.py", "tokenizer*", "*.md"])
    print(repo, "->", p)
PY
}

p_diff_heretic() {
  timeout 3600 "$PY" "$D/desktop/diff_public.py" \
      --base "$D/public/base" --other "$D/public/heretic-max" --limit 40
}
p_diff_ara() {
  timeout 3600 "$PY" "$D/desktop/diff_public.py" \
      --base "$D/public/base" --other "$D/public/ara" --limit 40
}

for step in p_pub_gate p_fetch p_diff_heretic p_diff_ara; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG7 COMPLETE"
