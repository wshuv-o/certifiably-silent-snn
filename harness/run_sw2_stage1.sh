#!/usr/bin/env bash
# SWEEP-002 stage 1: S0 control (teacher/init) and S1 scratch certificate loss. Seed 1.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models
( env SEED=1 TAG=S0_ctrl SAVE=$M/sw2_S0_ctrl_s1.pt python -u s4_improve.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)" ) &
( env SEED=1 TAG=S1_cert03 CERT_LAMBDA=0.3 SAVE=$M/sw2_S1_cert03_s1.pt python -u s4_improve.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 39)" ) &
wait
