#!/usr/bin/env bash
# Rerun of runs lost to CUDA_ERROR_UNKNOWN on 2026-10-06 (6 concurrent CUDA contexts in WSL).
# HARD CONSTRAINT: at most 3 concurrent CUDA processes. Measured: 3 jobs give ~1.9x throughput over 1,
# and 6 jobs gave NO additional throughput while causing driver faults. 3 is both faster and stable.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() {  # run <logprefix> <tag> <env...>
  pre=$1; tag=$2; shift 2
  env TEST=1 TAG=$tag "$@" SAVE=$M/${pre}_${tag}.pt python -u s4_improve.py > "$LOG/${pre}_$tag.log" 2>&1
  if grep -q "^RESULT" "$LOG/${pre}_$tag.log"; then grep -h "^RESULT" "$LOG/${pre}_$tag.log"
  else echo "*** FAILED: ${pre}_$tag ***"; tail -12 "$LOG/${pre}_$tag.log"; return 1; fi
}
echo "=== BATCH A: SCALE-FT-001 fine-tuned arms at H=1024 (3 concurrent) $(date +%H:%M:%S) ==="
export H=1024
for S in 1 2 3; do
  ( run sft ours_s$S SEED=$S INIT=$M/sft_ctrl_s$S.pt EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3 ) &
done
wait
echo "=== BATCH B: CONFIRM-002b seeds 6-7 at H=512 (2 chains) $(date +%H:%M:%S) ==="
unset H
chain() {
  S=$1
  run c2 ctrl_s$S SEED=$S || { echo "*** seed $S aborted: control failed ***"; return 1; }
  run c2 ours_s$S SEED=$S INIT=$M/c2_ctrl_s$S.pt EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3 || echo "*** seed $S ours failed ***"
  run c2 ref_cert03_s$S SEED=$S CERT_LAMBDA=0.3 || echo "*** seed $S ref failed ***"
}
for S in 6 7; do chain $S & done
wait
echo "=== RERUN COMPLETE $(date +%H:%M:%S) ==="
