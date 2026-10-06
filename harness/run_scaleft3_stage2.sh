#!/usr/bin/env bash
# SCALE-FT-003 stage 2 (pre-registered): corrected-rule winner = FT lambda 0.3, 40 ep, at H=1024,
# seeds 1-3, TEST SET, one shot. SERIAL (WSL GPU concurrency unreliable).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export H=1024
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() { pre=$1; tag=$2; shift 2
  echo "--- $tag $(date +%H:%M:%S) ---"
  env TEST=1 TAG=$tag "$@" SAVE=$M/${pre}_${tag}.pt python -u s4_improve.py > "$LOG/${pre}_$tag.log" 2>&1
  grep -q "^RESULT" "$LOG/${pre}_$tag.log" && echo "OK $tag" || { echo "*** FAILED: $tag ***"; tail -6 "$LOG/${pre}_$tag.log"; }
}
echo "=== SCALE-FT-003 STAGE 2 START $(date +%H:%M:%S) ==="
for S in 1 2 3; do
  C=$M/sft_ctrl_s$S.pt
  [ -f "$C" ] || { echo "*** missing $C ***"; continue; }
  run s3t ft40_l03_test_s$S INIT=$C LR=5e-4 RAMP=1 SEED=$S EPOCHS=40 CERT_LAMBDA=0.3
done
echo "=== STAGE 2 COMPLETE $(date +%H:%M:%S) ==="
