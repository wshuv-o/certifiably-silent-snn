#!/usr/bin/env bash
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
EPOCHS=1 SEED=1 NHUB=2 ALPHA=0.5 CERT_LAMBDA=0.3 TAG=smoke python -u s3_sweep.py 2>&1 | grep -vE "Warning|warn\(" | tail -5
rm -f ../results/sweep_smoke_s1.json