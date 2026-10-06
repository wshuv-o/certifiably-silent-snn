#!/usr/bin/env bash
# DELAY-002 (pre-registered in N3_SCALEUP_PLAN.md), two independent follow-ups to DELAY-001:
#  E1 = STACK the levers: H=1024 + delays + long schedule + strong augmentation. Best-accuracy attempt.
#  E2/E3 = MIN-DELAY test: DELAYS=2,4,8 has d_min=2, which by the delay decomposition gives 2 steps of
#          EXACT free lookahead (no bound needed at k<=2). Question: what does dropping the d=1 tap cost
#          in accuracy? If little, free lookahead is had for nearly nothing.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/d2_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/d2_${tag}.pt python -u s5_delays.py > "$LOG/d2_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/d2_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -12 "$LOG/d2_$tag.log"
}
echo "=== DELAY-002 START $(date +%H:%M:%S) ==="
( runone E2_mindelay2_40 H=512  SEED=1 EPOCHS=40  DELAYS=2,4,8 ) &
( runone E1_stack_h1024  H=1024 SEED=1 EPOCHS=150 DELAYS=1,2,4,8 AUG=2 ) &
wait
runone E3_mindelay2_long H=512 SEED=1 EPOCHS=150 DELAYS=2,4,8 AUG=2
echo "=== DELAY-002 COMPLETE $(date +%H:%M:%S) ==="
