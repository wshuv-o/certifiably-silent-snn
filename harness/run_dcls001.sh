#!/usr/bin/env bash
# DCLS-001 stage 1 (pre-registered): per-synapse learnable delays, seed 1, validation only.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
while pgrep -f "s5_delays.py" > /dev/null; do sleep 30; done
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/dc_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/dc_${tag}.pt python -u s7_dcls.py > "$LOG/dc_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/dc_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -15 "$LOG/dc_$tag.log"; return 1
}
BASE="H=512 SEED=1 DMIN=2 DMAX=8 AUG=2 EPOCHS=150"
echo "=== DCLS-001 START $(date +%H:%M:%S) ==="
runone DC1_ctrl $BASE || exit 1
runone DC1_cert $BASE INIT=$M/dc_DC1_ctrl.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0
echo "=== DCLS-001 COMPLETE $(date +%H:%M:%S) ==="
