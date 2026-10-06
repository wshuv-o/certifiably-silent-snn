#!/usr/bin/env bash
# PILOT-003b: same as PILOT-003 but with ring-local recurrent connectivity (pre-registered).
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
LOCAL=1 python -u pilot_silence.py 2>&1 | grep --line-buffered -vE "Warning|warn\("
df -h / | tail -1
