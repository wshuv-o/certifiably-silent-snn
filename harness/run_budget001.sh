#!/usr/bin/env bash
# BUDGET-001 (pre-registered in N3_SCALEUP_PLAN.md). Faster membrane leak (TAUM) and/or ring-local
# connectivity (LOCAL): do they cut the certification cost by changing R/budget instead of crushing
# weights? Each config gets its OWN matched control. Validation only, seed 1.
#
# Throughput design (measured on this machine):
#   1 proc = 4-6 s/epoch; 3 procs = 8 s/epoch each (~1.9x net throughput); 6 procs = flat AND unstable.
#   WSL2/Blackwell throws transient CUDA_ERROR_UNKNOWN and bogus OOMs under concurrency, which previously
#   destroyed 9 runs. So: cap at 3 concurrent, RETRY transient failures automatically, and SKIP work that
#   is already complete, so no failure or restart ever costs more than one run.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() {
  tag=$1; shift
  if grep -qs "^RESULT" "$LOG/b1_$tag.log" && [ -f "$M/b1_${tag}.pt" ]; then echo "SKIP $tag (already done)"; return 0; fi
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/b1_${tag}.pt python -u s4_improve.py > "$LOG/b1_$tag.log" 2>&1
    if grep -qs "^RESULT" "$LOG/b1_$tag.log" && [ -f "$M/b1_${tag}.pt" ]; then
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; fi
    echo "retry $tag: attempt $a failed ($(grep -oE 'CUDA error: [a-z ]+|OutOfMemoryError' "$LOG/b1_$tag.log" | head -1))"
    sleep 5
  done
  echo "*** FAILED after 3 attempts: $tag ***"; tail -8 "$LOG/b1_$tag.log"; return 1
}
chain() {  # id TAUM LOCAL  -- ours depends on this config's own control, so the chain is serial inside
  id=$1; tm=$2; lc=$3
  runone ${id}_ctrl SEED=1 TAUM=$tm LOCAL=$lc || { echo "*** $id aborted: control failed ***"; return 1; }
  runone ${id}_ours SEED=1 TAUM=$tm LOCAL=$lc INIT=$M/b1_${id}_ctrl.pt EPOCHS=40 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3
}
echo "=== BUDGET-001 START $(date +%H:%M:%S)  (3 concurrent chains, retry+resume) ==="
( chain B0 2.0 0 ) & ( chain B1 1.0 0 ) & ( chain B2 0.5 0 ) & wait
( chain B3 2.0 1 ) & ( chain B4 0.5 1 ) & wait
echo "=== BUDGET-001 COMPLETE $(date +%H:%M:%S) ==="
