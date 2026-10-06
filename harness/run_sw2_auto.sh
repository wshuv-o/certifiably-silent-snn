#!/usr/bin/env bash
# SWEEP-002: wait for S0 (teacher/init), run stage 2, then select the winner (validation only).
cd "$(dirname "$0")"
until [ -f ~/research/models/sw2_S0_ctrl_s1.pt ] && [ -f ../results/sw2_S1_cert03_s1.json ]; do sleep 20; done
bash run_sw2_stage2.sh
source ~/research/venv_gpu/bin/activate
python sw2_select.py
