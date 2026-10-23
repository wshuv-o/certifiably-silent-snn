#!/usr/bin/env bash
# NMNIST-001 (pre-registered as DVS-001, amended): does the mechanism survive a change of modality?
#
# SHD and SSC are both audio from the same pipeline, so the existing out-of-sample test never
# changes modality. N-MNIST is event-camera vision -- a DVS viewing MNIST digits through three
# saccades, 34x34 pixels and two polarities, ten classes.
#
# The SHD recipe is transferred with no tuning. Epochs are matched by GRADIENT STEPS, the same rule
# already applied to SSC: 58500 training samples after the monitor split give 457 steps per epoch,
# so 18 epochs is 8226 steps against SSC's 8250.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"

runone() {
  tag=$1; shift
  if grep -qas "^RESULT" "$LOG/nm_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/nm_${tag}.pt python -u s5_delays.py > "$LOG/nm_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/nm_$tag.log"; then
      sz=$(stat -c%s "$M/nm_${tag}.pt" 2>/dev/null || echo 0)
      if [ "$sz" -lt 10000 ]; then echo "*** $tag: RESULT but checkpoint $sz bytes ***"; return 1; fi
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag ***"; tail -25 "$LOG/nm_$tag.log"; return 1
}

BASE="DATASET=nmnist DELAYS=2,4,8 AUG=2 H=512 CPC=32 EPOCHS=18 TEST=1"
FT="LR=5e-4 RAMP=1 CERT_LAMBDA=1.0"

echo "=== NMNIST-001 START $(date) ==="
for S in 1 2 3; do
  runone ctrl_s$S $BASE SEED=$S                                   || exit 1
  runone cert_s$S $BASE SEED=$S INIT=$M/nm_ctrl_s$S.pt $FT        || exit 1
done
echo "=== NMNIST-001 COMPLETE $(date) ==="
