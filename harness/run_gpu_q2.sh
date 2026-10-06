#!/usr/bin/env bash
# GPU queue part 2: SSC-001 (15 epochs) then SCALE-001. Seed 1.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
run() { echo "=== $*"; env "$@" python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch)"; }
until [ -f ~/research/data/ssc/ssc_train.h5 ] && [ ! -f ~/research/data/ssc/ssc_train.h5.gz ]; do sleep 30; done
run DATASET=ssc EPOCHS=15 SEED=1 CERT_LAMBDA=0 SAVE=~/research/models/ssc_l0_s1.pt
run DATASET=ssc EPOCHS=15 SEED=1 CERT_LAMBDA=0.3 SAVE=~/research/models/ssc_l0.3_s1.pt