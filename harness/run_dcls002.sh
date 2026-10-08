#!/usr/bin/env bash
# DCLS-002 (pre-registered): does the architectural-certifiability claim survive a correct
# implementation of the delay learner?
#
# Our delays were trained at 1/100th the reference learning rate through a kernel too narrow for
# them to move. The claim that an unconstrained learnable-delay network certifies 97.4% of oracle
# may therefore be measuring the uniform initialisation rather than anything learned.
#
# Arm A reproduces the old result exactly (DCLS_FIX=0) so the comparison is like for like.
# Arm B is the corrected learner at the same delay range; arm C widens the range as the reference does.
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

BASE="H=512 SEED=1 AUG=2 EPOCHS=150"
FT="LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 EPOCHS=150"

echo "=== DCLS-002 START $(date) ==="

# B first: the corrected learner is the question; the old number (85.54%) is already logged in
# dc_DC1_ctrl.log, so reproducing it can wait until the end.
runone b_fix_ctrl  $BASE DMIN=2 DMAX=8  DCLS_FIX=1 || exit 1
runone b_fix_cert  H=512 SEED=1 AUG=2 DMIN=2 DMAX=8 DCLS_FIX=1 INIT=$M/dx_b_fix_ctrl.pt $FT || exit 1

# C: wider delay range, as the reference uses; the decomposition predicts this helps certifiability
runone c_wide_ctrl $BASE DMIN=2 DMAX=25 DCLS_FIX=1 || exit 1
runone c_wide_cert H=512 SEED=1 AUG=2 DMIN=2 DMAX=25 DCLS_FIX=1 INIT=$M/dx_c_wide_ctrl.pt $FT || exit 1

# A last: same code path as the original, kept only so the comparison is like for like on this machine
runone a_old_ctrl  $BASE DMIN=2 DMAX=8  DCLS_FIX=0 || exit 1

echo "=== DCLS-002 COMPLETE $(date) ==="
