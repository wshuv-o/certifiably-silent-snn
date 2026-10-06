#!/usr/bin/env bash
# DELAY-003 (pre-registered in N3_SCALEUP_PLAN.md): THE decisive test.
# E3 (H=512, DELAYS=2,4,8, 150 ep, AUG=2) is simultaneously the best-accuracy model (87.25 val) and the
# best-certifiability architecture (R_short 4.97; d_min=2 gives 2 steps of exact free lookahead).
# Question: constrained-fine-tuned from E3, can we get HIGH certification at ~87% accuracy -- i.e. does
# the accuracy/certifiability trade-off disappear entirely on this architecture?
# E3 itself is the matched control (identical architecture, no constraint).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
C=$M/d2_E3_mindelay2_long.pt
[ -f "$C" ] || { echo "*** ABORT: E3 checkpoint $C missing ***"; exit 1; }
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/d3_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/d3_${tag}.pt python -u s5_delays.py > "$LOG/d3_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/d3_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -12 "$LOG/d3_$tag.log"
}
BASE="H=512 SEED=1 DELAYS=2,4,8 AUG=2 INIT=$C EPOCHS=150 LR=5e-4 RAMP=1"
echo "=== DELAY-003 START $(date +%H:%M:%S) ==="
( runone F1_cert03 $BASE CERT_LAMBDA=0.3 ) &
( runone F2_cert10 $BASE CERT_LAMBDA=1.0 ) &
wait
# TAUM=0.5 raises the budget 2.2x, but needs its OWN matched control at this architecture
( runone F3_ctrl_taum05 H=512 SEED=1 DELAYS=2,4,8 AUG=2 EPOCHS=150 TAUM=0.5 ) &
wait
( runone F4_cert03_taum05 H=512 SEED=1 DELAYS=2,4,8 AUG=2 TAUM=0.5 INIT=$M/d3_F3_ctrl_taum05.pt EPOCHS=150 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3 ) &
wait
echo "=== DELAY-003 COMPLETE $(date +%H:%M:%S) ==="
