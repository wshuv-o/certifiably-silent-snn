#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
python -u fxp_check.py 2>&1 | grep --line-buffered -vE "Warning|warn\("