#!/usr/bin/env bash
# GOVERN-002 (pre-registered): rebuild the governing-relation table at a common training budget
# (150 epochs, AUG=2) across seeds 1-3. The published table mixed 40-epoch AUG=1 rows with
# 150-epoch AUG=2 rows, which confounds delay structure with training budget.
#
# Serial by design: this machine's GPU is unreliable with concurrent CUDA processes.
# Resumable: a run whose log already holds a RESULT line is skipped, so the script can be
# relaunched after a WSL restart without losing work.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"

runone() {           # runone <tag> <script> <env assignments...>
  tag=$1; script=$2; shift 2
  if grep -qas "^RESULT" "$LOG/g2_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/g2_${tag}.pt python -u "$script" > "$LOG/g2_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/g2_$tag.log"; then
      # a checkpoint that failed to write has bitten this project before; check it exists
      sz=$(stat -c%s "$(eval echo $M/g2_${tag}.pt)" 2>/dev/null || echo 0)
      if [ "$sz" -lt 10000 ]; then echo "*** $tag: RESULT but checkpoint is $sz bytes ***"; return 1; fi
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -20 "$LOG/g2_$tag.log"; return 1
}

COMMON="H=512 AUG=2 EPOCHS=150"
FT="LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 EPOCHS=150"

echo "=== GOVERN-002 START $(date) ==="
for S in 1 2 3; do
  # row 1: unit delay, the row the correction exists for -- no matched run exists at any seed
  runone u1_s$S  s5_delays.py $COMMON SEED=$S DELAYS=1       || exit 1
done

for S in 2 3; do     # seed 1 of the rows below already exists at the matched budget
  runone m4_s$S  s5_delays.py $COMMON SEED=$S DELAYS=1,2,4,8 || exit 1
  runone t3c_s$S s5_delays.py $COMMON SEED=$S DELAYS=2,4,8   || exit 1
  runone t3k_s$S s5_delays.py H=512 AUG=2 SEED=$S DELAYS=2,4,8 INIT=$M/g2_t3c_s$S.pt $FT || exit 1
  runone dcc_s$S s7_dcls.py   $COMMON SEED=$S DMIN=2 DMAX=8  || exit 1
  runone dck_s$S s7_dcls.py   H=512 AUG=2 SEED=$S DMIN=2 DMAX=8 INIT=$M/g2_dcc_s$S.pt $FT || exit 1
done
echo "=== GOVERN-002 COMPLETE $(date) ==="
