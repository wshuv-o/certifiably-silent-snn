#!/usr/bin/env bash
# S2-rep: seeds 2,3 at lambda in {0, 0.3} (pre-registered in research/N3_SCALEUP_PLAN.md)
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
for S in 2 3; do
  for L in 0 0.3; do
    echo "=== S2-rep SEED=$S CERT_LAMBDA=$L"
    CERT_LAMBDA=$L SEED=$S SAVE=~/research/models/s2_l${L}_s${S}.pt python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
  done
done