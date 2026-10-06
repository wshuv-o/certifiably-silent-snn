#!/usr/bin/env bash
# ALT-001 small model (CPU): seeds 1-3 x {clamp, l1, rate}; local ring, lambda=0 (alternatives replace the cert loss)
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
for S in 1 2 3; do
  for A in clamp l1 rate; do
    echo "=== ALT=$A SEED=$S"
    LOCAL=1 SEED=$S ALT=$A CERT_LAMBDA=0 SAVE=~/research/models/alt_${A}_s${S}.pt OUTJSON=../results/alt001_${A}_s${S}.json \
      python -u pilot_silence.py 2>&1 | grep --line-buffered -E "^(test accuracy|K=4|INTEGRITY)"
  done
done