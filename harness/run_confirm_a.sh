#!/usr/bin/env bash
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
run() { tag=$1; shift; echo "=== $tag $*"; env TEST=1 TAG=conf_$tag "$@" SAVE=~/research/models/conf_${tag}.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"; }
for S in 2 3 4; do run excap_s$S SEED=$S ALT=clamp; run ctrl_s$S SEED=$S; done