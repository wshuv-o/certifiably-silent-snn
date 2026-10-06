#!/usr/bin/env bash
# PILOT-005: seed replication of PILOT-004 (pre-registered). For each seed: lambda=0 baseline, lambda=0.1.
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
mkdir -p ~/research/models
for S in 1 2 3; do
  for L in 0 0.1; do
    echo "=== SEED=$S CERT_LAMBDA=$L"
    LOCAL=1 SEED=$S CERT_LAMBDA=$L SAVE=~/research/models/p005_s${S}_l${L}.pt python -u pilot_silence.py 2>&1 \
      | grep --line-buffered -E "^(test accuracy|K=4|INTEGRITY)"
  done
done
df -h / | tail -1
