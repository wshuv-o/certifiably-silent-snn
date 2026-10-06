#!/usr/bin/env bash
# Batch 1: finish CONFIRM-002b seeds 6-7 (lost to the CUDA incident).
# Batch 2: SCALE-FT-002 stage 1 -- fine-tuning-budget sweep at H=1024, VALIDATION ONLY (no TEST=1).
# HARD CONSTRAINT: <= 3 concurrent CUDA processes (6 caused CUDA_ERROR_UNKNOWN and gave no throughput).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() {
  pre=$1; tag=$2; shift 2
  env TAG=$tag "$@" SAVE=$M/${pre}_${tag}.pt python -u s4_improve.py > "$LOG/${pre}_$tag.log" 2>&1
  if grep -q "^RESULT" "$LOG/${pre}_$tag.log"; then grep -h "^RESULT" "$LOG/${pre}_$tag.log"
  else echo "*** FAILED: ${pre}_$tag ***"; tail -12 "$LOG/${pre}_$tag.log"; return 1; fi
}
echo "=== BATCH 1: CONFIRM-002b seeds 6-7 (H=512, test set) $(date +%H:%M:%S) ==="
chain() {
  S=$1
  run c2 ctrl_s$S TEST=1 SEED=$S || { echo "*** seed $S aborted: control failed ***"; return 1; }
  run c2 ours_s$S TEST=1 SEED=$S INIT=$M/c2_ctrl_s$S.pt EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3 || echo "*** seed $S ours failed ***"
  run c2 ref_cert03_s$S TEST=1 SEED=$S CERT_LAMBDA=0.3 || echo "*** seed $S ref failed ***"
}
for S in 6 7; do chain $S & done
wait
echo "=== BATCH 2: SCALE-FT-002 stage 1 (H=1024, VALIDATION ONLY, seed 1) $(date +%H:%M:%S) ==="
export H=1024
C=$M/sft_ctrl_s1.pt
[ -f "$C" ] || { echo "*** ABORT: $C missing ***"; exit 1; }
FTB="INIT=$C LR=5e-4 RAMP=1 SEED=1"
( run s2b ft1024_e20_l03 $FTB EPOCHS=20 CERT_LAMBDA=0.3 ) &
( run s2b ft1024_e40_l03 $FTB EPOCHS=40 CERT_LAMBDA=0.3 ) &
( run s2b ft1024_e20_l10 $FTB EPOCHS=20 CERT_LAMBDA=1.0 ) &
wait
( run s2b ft1024_e40_l10 $FTB EPOCHS=40 CERT_LAMBDA=1.0 ) &
wait
echo "=== BATCH COMPLETE $(date +%H:%M:%S) ==="
