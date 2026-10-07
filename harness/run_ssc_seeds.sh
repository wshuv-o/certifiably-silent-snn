#!/usr/bin/env bash
# SSC-FROZEN-001 seed extension: seeds 2-4 of the frozen recipe on SSC.
# Pre-registration: identical to SSC-FROZEN-001 (frozen SHD-winning recipe, zero retuning, epochs
# step-matched at 14). ALL seeds reported whatever they show; no stopping rule; the headline becomes
# the mean over seeds 1-4 with the seed-1 figure kept on record. Same anti-optional-stopping terms as
# CONFIRM-002b. Pass criteria unchanged: mean cost >= -1.0, certified >= 70% of oracle, 0 violations.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/ss_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/ss_${tag}.pt python -u s5_delays.py > "$LOG/ss_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/ss_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -12 "$LOG/ss_$tag.log"; return 1
}
FROZEN="DATASET=ssc H=512 DELAYS=2,4,8 AUG=2 TAUM=2.0 EPOCHS=14 TEST=1"
echo "=== SSC SEEDS 2-4 START $(date +%H:%M:%S) ==="
for S in 2 3 4; do
 (
  runone ssc_ctrl_s$S $FROZEN SEED=$S || { echo "*** seed $S aborted ***"; exit 1; }
  runone ssc_cert_s$S $FROZEN SEED=$S INIT=$M/ss_ssc_ctrl_s$S.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0
 ) &
done
wait
echo "=== SSC SEEDS COMPLETE $(date +%H:%M:%S) ==="
