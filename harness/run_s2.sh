#!/usr/bin/env bash
# S2 screening (seed 1): lambda in {0, 0.1, 0.3}, GPU. Pre-registered in research/N3_SCALEUP_PLAN.md
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
mkdir -p ~/research/models
for L in 0 0.1 0.3; do
  echo "=== S2 CERT_LAMBDA=$L"
  CERT_LAMBDA=$L SEED=1 SAVE=~/research/models/s2_l${L}_s1.pt python -u s2_strong.py 2>&1 | grep --line-buffered -vE "Warning|warn\("
done
