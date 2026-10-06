#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
python -u make_figures.py 2>&1 | grep -vE "Warning|warn\("
ls -la ../research/paper/figures/