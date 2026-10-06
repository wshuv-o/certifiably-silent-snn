#!/usr/bin/env bash
# DELAY-001 (pre-registered in N3_SCALEUP_PLAN.md). Multi-tap synaptic delays:
#   D0 = equivalence check (DELAYS=1 should behave like s4_improve.py, statistically)
#   D1 = does multi-tap delay raise accuracy at matched epochs?
#   D2 = best-effort accuracy (delays + long training + strong augmentation)
# Controls only (no constraint) -- first establish whether delays buy accuracy at all.
# Waits for the GPU, then runs SERIAL (delay taps multiply the per-step matmuls).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
while pgrep -f "s4_improve.py" > /dev/null; do sleep 20; done
runone() { tag=$1; shift
  if grep -qs "^RESULT" "$LOG/d1_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/d1_${tag}.pt python -u s5_delays.py > "$LOG/d1_$tag.log" 2>&1
    grep -qs "^RESULT" "$LOG/d1_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a: $(grep -oE 'Error[A-Za-z]*|CUDA error: [a-z ]+' "$LOG/d1_$tag.log" | head -1)"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -15 "$LOG/d1_$tag.log"
}
echo "=== DELAY-001 START $(date +%H:%M:%S) ==="
runone D0_d1        H=512 SEED=1 EPOCHS=40  DELAYS=1
runone D1_d1248     H=512 SEED=1 EPOCHS=40  DELAYS=1,2,4,8
runone D2_d1248_long H=512 SEED=1 EPOCHS=150 DELAYS=1,2,4,8 AUG=2
echo "=== DELAY-001 COMPLETE $(date +%H:%M:%S) ==="
