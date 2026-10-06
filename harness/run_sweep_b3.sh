#!/usr/bin/env bash
# SWEEP-001 batch 3 (pre-registered)
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
echo "=== ref_clamp"; env SEED=1 TAG=ref_clamp ALT=clamp SAVE=~/research/models/sweep_ref_clamp_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h1_l1_03"; env SEED=1 TAG=A_h1_l1_03 NHUB=1 ALT=l1 MU=0.3 SAVE=~/research/models/sweep_A_h1_l1_03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h2_l1_03"; env SEED=1 TAG=A_h2_l1_03 NHUB=2 ALT=l1 MU=0.3 SAVE=~/research/models/sweep_A_h2_l1_03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h4_l1_03"; env SEED=1 TAG=A_h4_l1_03 NHUB=4 ALT=l1 MU=0.3 SAVE=~/research/models/sweep_A_h4_l1_03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== C_a05_l01"; env SEED=1 TAG=C_a05_l01 ALPHA=0.5 CERT_LAMBDA=0.1 SAVE=~/research/models/sweep_C_a05_l01_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
