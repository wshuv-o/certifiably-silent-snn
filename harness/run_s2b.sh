#!/usr/bin/env bash
# S2b: stronger recipe (H=1024, learnable beta_i/rho_i, 60 epochs), lambda in {0, 0.3}, seed 1.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
run() { echo "=== $*"; env "$@" python -u s2b_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch)"; }
run SEED=1 CERT_LAMBDA=0 SAVE=~/research/models/s2b_l0_s1.pt
run SEED=1 CERT_LAMBDA=0.3 SAVE=~/research/models/s2b_l0.3_s1.pt