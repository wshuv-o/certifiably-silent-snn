#!/usr/bin/env bash
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
LOCAL=1 SEED=3 ALT=rate CERT_LAMBDA=0 SAVE=~/research/models/alt_rate_s3.pt OUTJSON=../results/alt001_rate_s3.json \
  python -u pilot_silence.py 2>&1 | grep --line-buffered -E "^(test accuracy|K=4|INTEGRITY)"