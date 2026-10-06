#!/usr/bin/env bash
# SHD-CONFIRM-FROZEN (pre-registered): frozen delay recipe, seeds 2-4, TEST SET evaluated once per run.
# Waits for the GPU so it does not contend with the SSC run.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
while pgrep -f "s5_delays.py" > /dev/null; do sleep 20; done
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/cf_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/cf_${tag}.pt python -u s5_delays.py > "$LOG/cf_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/cf_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -12 "$LOG/cf_$tag.log"; return 1
}
FROZEN="H=512 DELAYS=2,4,8 AUG=2 TAUM=2.0 EPOCHS=150 TEST=1"
echo "=== SHD-CONFIRM-FROZEN START $(date +%H:%M:%S) ==="
for S in 2 3 4; do
 (
  runone c_ctrl_s$S $FROZEN SEED=$S || { echo "*** seed $S aborted: control failed ***"; exit 1; }
  runone c_ours_s$S $FROZEN SEED=$S INIT=$M/cf_c_ctrl_s$S.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0
 ) &
done
wait
echo "=== SHD-CONFIRM-FROZEN COMPLETE $(date +%H:%M:%S) ==="
