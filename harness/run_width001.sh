#!/usr/bin/env bash
# WIDTH-001 (pre-registered in N3_SCALEUP_PLAN.md): does ring-local connectivity make certification
# width-independent? Cores hold a fixed 32 neurons, so LOCAL_R=1 gives fan-in 96 regardless of H --
# R should stay ~1 while capacity grows with H. Each config has its OWN matched control.
# <=3 concurrent CUDA processes, retry + resume (WSL/Blackwell throws transient CUDA faults).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() {
  tag=$1; shift
  if grep -qs "^RESULT" "$LOG/w1_$tag.log" && [ -f "$M/w1_${tag}.pt" ]; then echo "SKIP $tag"; return 0; fi
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/w1_${tag}.pt python -u s4_improve.py > "$LOG/w1_$tag.log" 2>&1
    if grep -qs "^RESULT" "$LOG/w1_$tag.log" && [ -f "$M/w1_${tag}.pt" ]; then
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; fi
    echo "retry $tag attempt $a: $(grep -oE 'CUDA error: [a-z ]+|OutOfMemoryError' "$LOG/w1_$tag.log" | head -1)"
    sleep 5
  done
  echo "*** FAILED after 3 attempts: $tag ***"; tail -8 "$LOG/w1_$tag.log"; return 1
}
chain() { id=$1; h=$2; tm=$3
  runone ${id}_ctrl H=$h SEED=1 TAUM=$tm LOCAL=1 || { echo "*** $id aborted ***"; return 1; }
  runone ${id}_ours H=$h SEED=1 TAUM=$tm LOCAL=1 INIT=$M/w1_${id}_ctrl.pt EPOCHS=40 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3
}
echo "=== WIDTH-001 START $(date +%H:%M:%S) ==="
( chain W1 1024 2.0 ) & ( chain W2 1024 0.5 ) & wait
( chain W3 2048 0.5 ) & wait
echo "=== WIDTH-001 COMPLETE $(date +%H:%M:%S) ==="
