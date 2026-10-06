#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
python pilot_edit_kv.py 2>&1 | grep -vE "Warning|warn\(|Loading weights|^\s*$"
df -h / | tail -1
