#!/usr/bin/env bash
# GPU queue part 3: SCALE-001 (H = 256, 1024), seed 1, SHD.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
run() { echo "=== $*"; env "$@" python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"; }
run SEED=1 H=256 CERT_LAMBDA=0
run SEED=1 H=256 CERT_LAMBDA=0.3
run SEED=1 H=1024 CERT_LAMBDA=0
run SEED=1 H=1024 CERT_LAMBDA=0.3