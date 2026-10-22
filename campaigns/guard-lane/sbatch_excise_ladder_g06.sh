#!/bin/bash
#SBATCH --job-name=fe-exlad06
#SBATCH --nodes=1
#SBATCH -p tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --export=ALL
#
# Guard lane, Qwen3Guard-Gen-0.6B (tag g06): pre-registered excise ladder with
# BOTH native reads on the 80/80 test bank. Grid and criteria are frozen in
# EXCISE-LADDER-g06-PREREG.md; do not edit the grid here without editing that.
#
# Model weights are already in the shared HF cache from the g06 harvest job, so
# offline mode is fine (compute nodes have no egress).
set -uo pipefail
source /scratch-shared/20234428/bullwhip/kit/config.sh
bw_load_env >/dev/null 2>&1
VENV=/scratch-shared/20234428/bullwhip/venv
export PYTHONPATH="$VENV/lib/python3.11/site-packages:$PYTHONPATH"
cd "$HOME/fenrir" || exit 1

GRID="26:0.75,26:1.0,26:1.1,26:1.25,26:1.5,24-26:1.0,24-26:1.1,24-26:1.25"
PAIRS="campaigns/guard-lane/data/guard_pairs_lfm-abl_slice2_n120.jsonl"

echo "[sbatch] node $(hostname) at $(date -u)"
echo "[sbatch] dry run first"
"$VENV/bin/python" campaigns/guard-lane/guard_excise_ladder.py \
    --pairs "$PAIRS" --dry-run || exit 1

echo "[sbatch] ladder: $GRID"
"$VENV/bin/python" campaigns/guard-lane/guard_excise_ladder.py \
    --config models/qwen3guard-0.6b.yaml \
    --pairs "$PAIRS" \
    --tag g06 \
    --grid "$GRID" \
    --weights o_proj,down_proj \
    --reads resp_mod,user_only \
    2>&1 | tail -60
rc=$?
echo "[sbatch] ladder exit $rc at $(date -u)"
echo "[sbatch] report: campaigns/guard-lane/results/excise_ladder_g06.json"
exit $rc
