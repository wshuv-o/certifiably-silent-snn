#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
PATTERN="~/research/models/alt_clamp_s{}.pt" OUTJSON=../results/fxp_excap.json python -u fxp_check.py 2>&1 | grep -vE "Warning|warn\("
