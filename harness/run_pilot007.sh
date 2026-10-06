#!/usr/bin/env bash
# PILOT-007: sparse-regime test (pre-registered). Seed 1; rate target 1%;
# V1 = 8 cores x 32, V2 = 16 cores x 16; lambda in {0, 0.1}; then synchronization-message counts.
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
M=~/research/models
for NPV in 8 16; do
  for L in 0 0.1; do
    echo "=== NP=$NPV RATE_TARGET=0.01 CERT_LAMBDA=$L"
    LOCAL=1 SEED=1 NP=$NPV RATE_TARGET=0.01 CERT_LAMBDA=$L SAVE=$M/p007_np${NPV}_l${L}.pt \
      python -u pilot_silence.py 2>&1 | grep --line-buffered -E "^(test accuracy|K=4|INTEGRITY)"
  done
done
for NPV in 8 16; do
  echo "=== SYNC MESSAGES NP=$NPV"
  NP=$NPV PAIRS="np${NPV}=$M/p007_np${NPV}_l0.pt,$M/p007_np${NPV}_l0.1.pt" OUT=../results/pilot007_np${NPV}.json \
    python -u pilot_sync_msgs.py 2>&1 | grep --line-buffered -vE "Warning|warn\("
done
df -h / | tail -1
