#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
python -c "import h5py" 2>/dev/null || pip install -q --no-cache-dir h5py
cd "$(dirname "$0")"
python pilot_silence.py 2>&1 | grep -vE "Warning|warn\("
df -h / | tail -1
