#!/usr/bin/env bash
# PILOT-004 integrity check: rerun lambda=0.1 (same seed), save model, run soundness / non-triviality checks.
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
mkdir -p ~/research/models
LOCAL=1 CERT_LAMBDA=0.1 SAVE=~/research/models/pilot004_l0.1.pt python -u pilot_silence.py 2>&1 \
  | grep --line-buffered -vE "Warning|warn\("
df -h / | tail -1
