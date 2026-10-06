#!/usr/bin/env bash
# SWEEP-002 stage 2: fine-tuning / distillation configs (need S0). Seed 1. Two parallel queues.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models; T0=$M/sw2_S0_ctrl_s1.pt
FT="INIT=$T0 EPOCHS=20 LR=5e-4 RAMP=1"
run() { tag=$1; shift; env SEED=1 TAG=$tag "$@" SAVE=$M/sw2_${tag}_s1.pt python -u s4_improve.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch 19|epoch 39)"; }
( run S2_ft_cert03 $FT CERT_LAMBDA=0.3; run S3_ft_cert03_kd $FT CERT_LAMBDA=0.3 TEACHER=$T0; run S4_ft_cert10_kd $FT CERT_LAMBDA=1.0 TEACHER=$T0 ) &
( run S6_ft_hub4_kd $FT NHUB=4 CERT_LAMBDA=0.3 TEACHER=$T0; run S7_ft_excap_kd $FT ALT=clamp TEACHER=$T0; run S5_scratch_cert03_kd CERT_LAMBDA=0.3 TEACHER=$T0 ) &
wait
