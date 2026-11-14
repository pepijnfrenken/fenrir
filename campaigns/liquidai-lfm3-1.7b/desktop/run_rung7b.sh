#!/usr/bin/env bash
# LFM3-1.7B — rung 7b: the same public build, at a window that clears its
# thinking block, plus the weight-space recovery that rung 7 never reached.
#
# Why this rung exists: at 1024 tokens the public build read 0/12 refusals and
# 12/12 coherent, and EVERY one of the 12 responses hit the cap inside its
# reasoning block with no closing think tag. "No refusal" is therefore measured;
# "answers or slop" is not. The window is the defect, not the model. 4096 tokens
# is ~2.4x the observed reasoning length.
#
# Also: `collect` refuses to certify a model_id it treats as pristine when that
# model does not refuse (the baseline sanity gate, correctly). A foreign build
# has to be passed as --model-dir so the guard does not confuse "this checkpoint
# does not refuse" with "our instrument is blind".
#
# Output handling: a genuinely uncensored 1.7B will emit actionable harmful
# content at 4096 tokens. Raw generations stay on this desktop
# (campaigns/.../public-transcripts/, git-ignored). Only counts, label tables and
# 60-character heads are ever committed.
set -uo pipefail
REPO="$HOME/absolver"; D="$REPO/campaigns/liquidai-lfm3-1.7b"; L="$D/desktop/logs"
PY="$REPO/.venv/bin/python"
export PYTHONUNBUFFERED=1 HF_HOME="$HOME/hf" TOKENIZERS_PARALLELISM=false
cd "$REPO" || exit 5; mkdir -p "$L" "$D/public" "$D/public-transcripts"

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

p_fetch_public() {
  "$PY" - <<'PY'
from huggingface_hub import snapshot_download
import os
root = os.path.expanduser("~/absolver/campaigns/liquidai-lfm3-1.7b/public/heretic-max")
p = snapshot_download(repo_id="MihaiPopa-1/LFM3-1.7B-Heretic-Max", local_dir=root,
                      allow_patterns=["*.safetensors", "*.json", "*.py", "tokenizer*"])
print("mihai ->", p)
PY
}

p_gate_4096() {
  # foreign build as --model-dir: no pristine fail-fast, gates still measured
  timeout 14400 "$PY" harness/abl.py collect models/lfm3-1.7b-pub-w4096.yaml \
      --model-dir "$D/public/heretic-max" --transcript
}

p_judge_4096() {
  local t
  t=$(find "$D" -path "*heretic-max*" -name transcript.json | head -1)
  [ -z "$t" ] && { echo "no transcript"; return 3; }
  echo "judging $t"
  cp "$t" "$D/public-transcripts/heretic-max-w4096-transcript.json"
  timeout 2400 "$PY" "$D/desktop/judge_transcript.py" "$t" --out "$D/judge-r7b-heretic-w4096.jsonl"
}

p_fetch_base() {
  "$PY" - <<'PY'
from huggingface_hub import snapshot_download
import os
for repo, dest in [("LiquidAI/LFM3-1.7B", "public/base"),
                   ("Austriani/LFM3-1.7B-ARA-Abliterated", "public/ara")]:
    root = os.path.expanduser(f"~/absolver/campaigns/liquidai-lfm3-1.7b/{dest}")
    p = snapshot_download(repo_id=repo, local_dir=root,
                          allow_patterns=["*.safetensors", "*.json", "*.py", "tokenizer*"])
    print(repo, "->", p)
PY
}

p_diff_heretic() {
  timeout 3600 "$PY" "$D/desktop/diff_public.py" --base "$D/public/base" \
      --other "$D/public/heretic-max" --limit 40
}
p_diff_ara() {
  timeout 3600 "$PY" "$D/desktop/diff_public.py" --base "$D/public/base" \
      --other "$D/public/ara" --limit 40
}

for step in p_fetch_public p_gate_4096 p_judge_4096 p_fetch_base p_diff_heretic p_diff_ara; do
  phase "${step#p_}" "$step"; rc=$?
  [ "$rc" -ne 0 ] && { echo "!! stop at $step rc=$rc"; exit "$rc"; }
done
echo "== RUNG7B COMPLETE"
