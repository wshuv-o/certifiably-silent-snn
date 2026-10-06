#!/usr/bin/env bash
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
DATASET=ssc EPOCHS=15 SEED=1 ALT=l1 CERT_LAMBDA=0 SAVE=~/research/models/ssc_l1_s1.pt \
  python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch)"