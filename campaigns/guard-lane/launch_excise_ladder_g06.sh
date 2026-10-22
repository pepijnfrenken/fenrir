#!/bin/bash
# Launch the pre-registered g06 excise ladder on the TU/e Umbrella cluster from
# the hub. Copies the driver + sbatch script to the cluster repo and submits.
#
#   bash ~/absolver/campaigns/guard-lane/launch_excise_ladder_g06.sh
#
# Preconditions: the WireGuard tunnel is UP (ping 131.155.7.108 answers) and the
# g06 direction file exists on the cluster
# (~/fenrir/campaigns/guard-lane/results/directions_g06.npz).
set -uo pipefail

LOCAL="$HOME/absolver"
REMOTE="$HOME/fenrir"
HOST="tue-cluster"
FILES=(
  campaigns/guard-lane/guard_excise_ladder.py
  campaigns/guard-lane/test_excise_ladder_offline.py
  campaigns/guard-lane/sbatch_excise_ladder_g06.sh
  campaigns/guard-lane/EXCISE-LADDER-g06-PREREG.md
)

echo "[launch] tunnel check"
if ! ping -c 2 -W 3 131.155.7.108 >/dev/null 2>&1; then
  echo "[launch] ABORT: 131.155.7.108 does not answer. Renew the eduVPN profile and"
  echo "         bring the tunnel up first (sudo wg-quick down tue && sudo wg-quick up tue)."
  exit 1
fi

echo "[launch] preconditions on the cluster"
ssh -o ConnectTimeout=15 "$HOST" "
  ls -l '$REMOTE/campaigns/guard-lane/results/directions_g06.npz' || exit 2
  ls -d /scratch-shared/20234428/bullwhip/hf/hub/models--Qwen--Qwen3Guard-Gen-0.6B* >/dev/null || \
    echo '[launch] WARN: 0.6B guard not in the shared HF cache; stage it on the LOGIN node first'
  sacct -j 0 >/dev/null 2>&1; echo '[launch] cluster reachable'
" || exit 2

echo "[launch] copying ${#FILES[@]} files"
for f in "${FILES[@]}"; do
  scp -o ConnectTimeout=15 "$LOCAL/$f" "$HOST:$REMOTE/$f" || exit 3
done

echo "[launch] smoke (offline self-test) on the cluster"
ssh -o ConnectTimeout=20 "$HOST" "
  source /scratch-shared/20234428/bullwhip/kit/config.sh >/dev/null 2>&1
  bw_load_env >/dev/null 2>&1
  export PYTHONPATH=/scratch-shared/20234428/bullwhip/venv/lib/python3.11/site-packages:\$PYTHONPATH
  cd $REMOTE && /scratch-shared/20234428/bullwhip/venv/bin/python \
    campaigns/guard-lane/test_excise_ladder_offline.py 2>&1 | tail -12
" || exit 4

echo "[launch] submitting"
JID=$(ssh -o ConnectTimeout=20 "$HOST" "cd $REMOTE && sbatch --parsable campaigns/guard-lane/sbatch_excise_ladder_g06.sh") || exit 5
echo "[launch] job $JID submitted"
echo "[launch] watch with: ssh $HOST 'sacct -j $JID --noheader --format=State'"
echo "[launch] report lands at: $REMOTE/campaigns/guard-lane/results/excise_ladder_g06.json"
