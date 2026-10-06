#!/usr/bin/env bash
# SWEEP-001 batch 0 (pre-registered)
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
echo "=== ref_ctrl"; env SEED=1 TAG=ref_ctrl  SAVE=~/research/models/sweep_ref_ctrl_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h1_cert03"; env SEED=1 TAG=A_h1_cert03 NHUB=1 CERT_LAMBDA=0.3 SAVE=~/research/models/sweep_A_h1_cert03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h2_cert03"; env SEED=1 TAG=A_h2_cert03 NHUB=2 CERT_LAMBDA=0.3 SAVE=~/research/models/sweep_A_h2_cert03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== A_h4_cert03"; env SEED=1 TAG=A_h4_cert03 NHUB=4 CERT_LAMBDA=0.3 SAVE=~/research/models/sweep_A_h4_cert03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== C_a0_l01"; env SEED=1 TAG=C_a0_l01 ALPHA=0.0 CERT_LAMBDA=0.1 SAVE=~/research/models/sweep_C_a0_l01_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
echo "=== C_a05_l03"; env SEED=1 TAG=C_a05_l03 ALPHA=0.5 CERT_LAMBDA=0.3 SAVE=~/research/models/sweep_C_a05_l03_s1.pt python -u s3_sweep.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)"
