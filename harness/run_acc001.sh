#!/usr/bin/env bash
# ACC-001 (pre-registered in N3_SCALEUP_PLAN.md): close the accuracy gap to the published 95.07% by
# adopting the reference's training recipe, while keeping our recurrent architecture.
#
# The gap is 87.25 vs 95.07. Two causes were already fixed (delay learning rate, kernel annealing).
# The rest is recipe: dropout, batch normalisation, one-cycle LR, batch 256, ATan surrogate, input
# binning, and a fine-tuning tail. Arms are added one item at a time so each is attributable, but the
# full recipe runs first: if it does not move accuracy there is nothing to attribute.
#
# Validation only. No TEST=1 anywhere in this file. The test set is touched once, after the recipe is
# frozen, by a separate confirmation run.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"

runone() {
  tag=$1; shift
  if grep -qas "^RESULT" "$LOG/ac_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=ac_$tag "$@" SAVE=$M/ac_${tag}.pt python -u s5_delays.py > "$LOG/ac_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/ac_$tag.log"; then
      sz=$(stat -c%s "$M/ac_${tag}.pt" 2>/dev/null || echo 0)
      [ "$sz" -lt 10000 ] && { echo "*** $tag: RESULT but checkpoint $sz bytes ***"; return 1; }
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag ***"; tail -25 "$LOG/ac_$tag.log"; return 1
}

BASE="H=512 SEED=1 AUG=2 DELAYS=2,4,8 EPOCHS=150"
FULL="DROP=0.4 BN=1 ONECYCLE=1 BS=256 SURR=atan NBINS=5 FT_EP=30"

echo "=== ACC-001 START $(date) ==="

case "${ARM:-both}" in
  full) runone r_full $BASE $FULL ;;
  base) runone r_base $BASE ;;       # anchor: the current recipe on this exact code path
  *)    runone r_full $BASE $FULL; runone r_base $BASE ;;
esac

echo "=== ACC-001 ARM ${ARM:-both} COMPLETE $(date) ==="
