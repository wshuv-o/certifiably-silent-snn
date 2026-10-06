#!/usr/bin/env bash
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
run() { echo "=== $*"; env "$@" python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"; }
for S in 2 3; do run SEED=$S H=1024 CERT_LAMBDA=0; run SEED=$S H=1024 CERT_LAMBDA=0.3; done