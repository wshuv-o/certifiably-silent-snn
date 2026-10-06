#!/usr/bin/env bash
# ACC-001: how much of the 77.4 -> 96 accuracy gap is just undertraining / weak regularisation,
# before building delays and depth? Controls only (no constraint), dense H=1024, validation only.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() { tag=$1; shift
  if grep -qs "^RESULT" "$LOG/a1_$tag.log"; then echo "SKIP $tag"; return 0; fi
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/a1_${tag}.pt python -u s4_improve.py > "$LOG/a1_$tag.log" 2>&1
    grep -qs "^RESULT" "$LOG/a1_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -6 "$LOG/a1_$tag.log"
}
echo "=== ACC-001 START $(date +%H:%M:%S) ==="
( runone A1_ep150      H=1024 SEED=1 EPOCHS=150 ) &
( runone A2_ep150_aug2 H=1024 SEED=1 EPOCHS=150 AUG=2 ) &
( runone A3_ep300_aug2 H=1024 SEED=1 EPOCHS=300 AUG=2 ) &
wait
echo "=== ACC-001 COMPLETE $(date +%H:%M:%S) ==="
