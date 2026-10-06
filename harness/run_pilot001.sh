#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
python pilot_encoder_change.py "$@" 2>&1 | grep -vE "Warning|warn\(|^\s*$"
df -h / | tail -1
