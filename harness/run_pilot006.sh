#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
python -u pilot_sync_msgs.py 2>&1 | grep --line-buffered -vE "Warning|warn\("
