#!/usr/bin/env bash
# DELAY-004: push the best configuration further on the two levers already measured to work.
#   A3 beat A2 by +1.2 pts purely from 300 vs 150 epochs, so train the E3/F2 architecture longer.
#   G1 = unconstrained control at 300 ep (matched control for G2)
#   G2 = constrained (cert lambda 1.0, the stronger constraint that is currently AHEAD) at 300 ep
# Waits for DELAY-003 to finish so CPU/GPU contention does not distort anything.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
# (serialised by run_overnight.sh)
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/d4_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/d4_${tag}.pt python -u s5_delays.py > "$LOG/d4_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/d4_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -12 "$LOG/d4_$tag.log"
}
echo "=== DELAY-004 START $(date +%H:%M:%S) ==="
runone G1_ctrl300 H=512 SEED=1 DELAYS=2,4,8 AUG=2 EPOCHS=300
runone G2_cert10_300 H=512 SEED=1 DELAYS=2,4,8 AUG=2 INIT=$M/d4_G1_ctrl300.pt EPOCHS=300 LR=5e-4 RAMP=1 CERT_LAMBDA=1.0
echo "=== DELAY-004 COMPLETE $(date +%H:%M:%S) ==="
