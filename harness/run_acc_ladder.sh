#!/usr/bin/env bash
# ACC-001 stage 2: attribute the recipe, by leave-one-out from the full recipe.
#
# Stage 1 ran the reference's full recipe on our recurrent model and it UNDERPERFORMED the current
# one: validation accuracy plateaued near 0.80 and drifted down over the last twenty epochs, while the
# unchanged recipe was already at 0.82 by epoch 64 and heads for ~0.87. So at least one ingredient
# that helps the reference's feedforward delay architecture hurts ours.
#
# With the full recipe broken, leave-one-out is the informative design: each arm removes exactly one
# ingredient, so an arm that recovers accuracy names the culprit. Adding one at a time to the base
# would instead measure six near-null effects before finding it.
#
# Suspects in order. Input binning collapses 700 channels to 140, which the reference can afford
# because it has two 256-unit layers and a feedforward delay stack to recover frequency structure,
# and we have one recurrent layer. Dropout at 0.4 is heavy for 512 units. Batch normalisation of the
# feedforward current interacts with a recurrence it cannot normalise.
#
# Validation only; no test evaluation.
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
    grep -qas "^RESULT" "$LOG/ac_$tag.log" && { echo "OK $tag $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag ***"; tail -20 "$LOG/ac_$tag.log"; return 1
}

BASE="H=512 SEED=1 AUG=2 DELAYS=2,4,8 EPOCHS=150"

case "${ARM:?set ARM}" in
  # each line is the full recipe with exactly one ingredient removed
  nobins) runone r_nobins $BASE DROP=0.4 BN=1 ONECYCLE=1 BS=256 SURR=atan          FT_EP=30 ;;
  nodrop) runone r_nodrop $BASE         BN=1 ONECYCLE=1 BS=256 SURR=atan NBINS=5   FT_EP=30 ;;
  nobn)   runone r_nobn   $BASE DROP=0.4      ONECYCLE=1 BS=256 SURR=atan NBINS=5  FT_EP=30 ;;
  nooc)   runone r_nooc   $BASE DROP=0.4 BN=1            BS=256 SURR=atan NBINS=5  FT_EP=30 ;;
  # and the cheapest plausible win on its own, to check the surrogate in isolation
  atan)   runone r_atan   $BASE SURR=atan ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
