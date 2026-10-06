#!/usr/bin/env bash
# PILOT-004: certified-silence training, local connectivity, lambda in {0.1, 1.0} (pre-registered).
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
for L in 0.1 1.0; do
  echo "=== CERT_LAMBDA=$L"
  LOCAL=1 CERT_LAMBDA=$L python -u pilot_silence.py 2>&1 | grep --line-buffered -vE "Warning|warn\("
done
df -h / | tail -1
