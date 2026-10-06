#!/usr/bin/env bash
# SWEEP-001 batch 2 (pre-registered)
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
echo "=== ref_l1"; env SEED=1 TAG=ref_l1 ALT=l1 MU=0.1 SAVE=~/research/models/sweep_ref_l1_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h1_l1_01"; env SEED=1 TAG=A_h1_l1_01 NHUB=1 ALT=l1 MU=0.1 SAVE=~/research/models/sweep_A_h1_l1_01_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h2_l1_01"; env SEED=1 TAG=A_h2_l1_01 NHUB=2 ALT=l1 MU=0.1 SAVE=~/research/models/sweep_A_h2_l1_01_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h4_l1_01"; env SEED=1 TAG=A_h4_l1_01 NHUB=4 ALT=l1 MU=0.1 SAVE=~/research/models/sweep_A_h4_l1_01_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== C_a0_l10"; env SEED=1 TAG=C_a0_l10 ALPHA=0.0 CERT_LAMBDA=1.0 SAVE=~/research/models/sweep_C_a0_l10_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
