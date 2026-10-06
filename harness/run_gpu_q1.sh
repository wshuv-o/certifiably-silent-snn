#!/usr/bin/env bash
# GPU queue part 1 (pre-registered ALT-001 strong + LAMBDA-001 strong). Seed 1, SHD.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
run() { echo "=== $*"; env "$@" python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"; }
run SEED=1 ALT=clamp CERT_LAMBDA=0
run SEED=1 ALT=l1 CERT_LAMBDA=0
run SEED=1 CERT_LAMBDA=0.03
run SEED=1 CERT_LAMBDA=1.0