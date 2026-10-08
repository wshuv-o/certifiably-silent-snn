#!/usr/bin/env bash
# Seeds 2 and 3 for the corrected delay learner.
#
# Table 1 compares six architectures at a common budget over three seeds each. Its learnable-delay
# rows were produced by a learner that could not move its delays (one learning rate for positions and
# weights, and a kernel too narrow to carry gradient between taps), so they describe the uniform
# initialisation rather than a learned delay distribution. DCLS-002 replaced them on seed 1; these two
# runs make the replacement comparable to the rest of the table.
#
# Validation only. No test evaluation here.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"

runone() {
  tag=$1; shift
  if grep -qas "^RESULT" "$LOG/dx_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/dx_${tag}.pt python -u s7_dcls.py > "$LOG/dx_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/dx_$tag.log"; then
      sz=$(stat -c%s "$M/dx_${tag}.pt" 2>/dev/null || echo 0)
      [ "$sz" -lt 10000 ] && { echo "*** $tag: RESULT but checkpoint $sz bytes ***"; return 1; }
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag ***"; tail -25 "$LOG/dx_$tag.log"; return 1
}

S=${S:-2}
echo "=== corrected learner, seed $S, START $(date) ==="
runone b_fix_ctrl_s$S H=512 SEED=$S AUG=2 EPOCHS=150 DMIN=2 DMAX=8 DCLS_FIX=1 || exit 1
runone b_fix_cert_s$S H=512 SEED=$S AUG=2 DMIN=2 DMAX=8 DCLS_FIX=1 \
       INIT=$M/dx_b_fix_ctrl_s$S.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 EPOCHS=150 || exit 1
echo "=== seed $S COMPLETE $(date) ==="
