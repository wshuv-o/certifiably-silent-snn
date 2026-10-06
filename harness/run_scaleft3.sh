#!/usr/bin/env bash
# SCALE-FT-003 (pre-registered in N3_SCALEUP_PLAN.md): replicate the two candidate 1024 configs on
# VALIDATION seeds 2-3. SERIAL (WSL GPU concurrency is unreliable). No test evaluation here.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export H=1024
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() { pre=$1; tag=$2; shift 2
  echo "--- $tag $(date +%H:%M:%S) ---"
  env TAG=$tag "$@" SAVE=$M/${pre}_${tag}.pt python -u s4_improve.py > "$LOG/${pre}_$tag.log" 2>&1
  grep -q "^RESULT" "$LOG/${pre}_$tag.log" && echo "OK $tag" || { echo "*** FAILED: $tag ***"; tail -6 "$LOG/${pre}_$tag.log"; }
}
echo "=== SCALE-FT-003 START $(date +%H:%M:%S) ==="
for S in 2 3; do
  C=$M/sft_ctrl_s$S.pt
  [ -f "$C" ] || { echo "*** missing $C ***"; continue; }
  run s3v ft1024_e40_l03_s$S INIT=$C LR=5e-4 RAMP=1 SEED=$S EPOCHS=40 CERT_LAMBDA=0.3
  run s3v ft1024_e20_l10_s$S INIT=$C LR=5e-4 RAMP=1 SEED=$S EPOCHS=20 CERT_LAMBDA=1.0
done
echo "=== SCALE-FT-003 COMPLETE $(date +%H:%M:%S) ==="
