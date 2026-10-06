#!/usr/bin/env bash
# SERIAL rerun of everything lost to WSL GPU instability.
# Concurrency is unreliable on this setup at ANY level: 6 procs -> CUDA_ERROR_UNKNOWN; 2-3 procs ->
# CUDA_ERROR_UNKNOWN *and* a bogus OOM (48 MiB alloc refused with 10.45 GiB free, while nvidia/WSL
# reported "17179869184 GiB in use" = 2^34, i.e. corrupted GPU memory accounting in WSL2).
# Since each failure costs a full rerun, serial is faster in expectation. One process at a time.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() {
  pre=$1; tag=$2; shift 2
  echo "--- $tag  $(date +%H:%M:%S) ---"
  env TAG=$tag "$@" SAVE=$M/${pre}_${tag}.pt python -u s4_improve.py > "$LOG/${pre}_$tag.log" 2>&1
  if grep -q "^RESULT" "$LOG/${pre}_$tag.log"; then echo "OK $tag"
  else echo "*** FAILED: $tag ***"; tail -6 "$LOG/${pre}_$tag.log"; fi
}
echo "=== SERIAL PENDING START $(date +%H:%M:%S) ==="
# 1) CONFIRM-002b: the one missing reference arm (H=512, test set)
run c2 ref_cert03_s7 TEST=1 SEED=7 CERT_LAMBDA=0.3
# 2) SCALE-FT-002 stage 1: the three lost configs (H=1024, VALIDATION ONLY, seed 1)
export H=1024
C=$M/sft_ctrl_s1.pt
FTB="INIT=$C LR=5e-4 RAMP=1 SEED=1"
run s2b ft1024_e20_l03 $FTB EPOCHS=20 CERT_LAMBDA=0.3
run s2b ft1024_e40_l03 $FTB EPOCHS=40 CERT_LAMBDA=0.3
run s2b ft1024_e20_l10 $FTB EPOCHS=20 CERT_LAMBDA=1.0
echo "=== SERIAL PENDING COMPLETE $(date +%H:%M:%S) ==="
