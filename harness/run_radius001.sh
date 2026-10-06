#!/usr/bin/env bash
# RADIUS-001 (pre-registered in N3_SCALEUP_PLAN.md): is there a connectivity radius that keeps
# certification high and free while recovering the dense model's accuracy?
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() { tag=$1; shift
  if grep -qs "^RESULT" "$LOG/r1_$tag.log" && [ -f "$M/r1_${tag}.pt" ]; then echo "SKIP $tag"; return 0; fi
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/r1_${tag}.pt python -u s4_improve.py > "$LOG/r1_$tag.log" 2>&1
    if grep -qs "^RESULT" "$LOG/r1_$tag.log" && [ -f "$M/r1_${tag}.pt" ]; then echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; fi
    echo "retry $tag attempt $a: $(grep -oE 'CUDA error: [a-z ]+|OutOfMemoryError' "$LOG/r1_$tag.log" | head -1)"; sleep 5
  done
  echo "*** FAILED after 3: $tag ***"; tail -8 "$LOG/r1_$tag.log"; return 1
}
chain() { r=$1
  runone R${r}_ctrl H=1024 SEED=1 TAUM=0.5 LOCAL=1 LOCAL_R=$r || { echo "*** R$r aborted ***"; return 1; }
  runone R${r}_ours H=1024 SEED=1 TAUM=0.5 LOCAL=1 LOCAL_R=$r INIT=$M/r1_R${r}_ctrl.pt EPOCHS=40 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3
}
echo "=== RADIUS-001 START $(date +%H:%M:%S) ==="
for r in 2 4 8; do ( chain $r ) & done
wait
echo "=== RADIUS-001 COMPLETE $(date +%H:%M:%S) ==="
