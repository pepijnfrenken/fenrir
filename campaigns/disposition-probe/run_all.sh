#!/usr/bin/env bash
# Rung 0: disposition probe, three arms, identical item grid.
# Bounds: 1 download (~2.3 GB each arm, cached), 1 extra checkpoint written,
# ~120 greedy generations per arm. Cleanup: see campaigns/disposition-probe/README.md
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=~/absolver/.model-cache
PY=.venv/bin/python
D=campaigns/disposition-probe
N=${N:-120}

mkdir -p "$D/models" "$D/results"

if [ ! -f "$D/models/noise-0.025/config.json" ]; then
  echo "== building noise control (rel_l2 0.025)"
  $PY $D/build_noise_control.py --model LiquidAI/LFM2.5-1.2B-Instruct \
      --out "$D/models/noise-0.025" --rel 0.025
fi

for arm in pristine abliterated noise; do
  case $arm in
    pristine)     M=LiquidAI/LFM2.5-1.2B-Instruct ;;
    abliterated)  M=PinoCookie/LFM2.5-1.2B-Instruct-Abliterated ;;
    noise)        M=$D/models/noise-0.025 ;;
  esac
  OUT=$D/results/$arm.jsonl
  if [ -s "$OUT" ]; then echo "== $arm already has results, skipping"; continue; fi
  echo "== $arm ($M)"
  $PY $D/probe_disposition.py --model "$M" --arm "$arm" --n "$N" \
      --mode prob --out "$OUT"
done

echo "== analysing"
cat "$D/results/pristine.jsonl" "$D/results/abliterated.jsonl" \
    "$D/results/noise.jsonl" > "$D/results/combined.jsonl"
$PY $D/analyze.py "$D/results/combined.jsonl" | tee "$D/results/summary.txt"
echo "== done"
