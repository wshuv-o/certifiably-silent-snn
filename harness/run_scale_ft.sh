#!/usr/bin/env bash
# SCALE-FT-001 (pre-registered in research/N3_SCALEUP_PLAN.md): does the CONFIRM-002 fine-tuning recipe
# remove the accuracy cost at H=1024, where from-scratch training cost -2.1 points?
# Full logging, no grep pipe (see run_sw2_stage1.sh note).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export H=1024
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() {
  tag=$1; shift
  env TEST=1 TAG=$tag "$@" SAVE=$M/sft_${tag}.pt python -u s4_improve.py > "$LOG/sft_$tag.log" 2>&1
  if grep -q "^RESULT" "$LOG/sft_$tag.log"; then grep -h "^RESULT" "$LOG/sft_$tag.log"
  else echo "*** FAILED: $tag ***"; tail -25 "$LOG/sft_$tag.log"; return 1; fi
  [ -f "$M/sft_${tag}.pt" ] || { echo "*** MISSING CHECKPOINT: sft_${tag}.pt ***"; return 1; }
}
chain() {
  S=$1; CTRL=$M/sft_ctrl_s$S.pt
  run ctrl_s$S SEED=$S || { echo "*** seed $S aborted: control failed ***"; return 1; }
  run ours_s$S SEED=$S INIT=$CTRL EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3 || echo "*** seed $S: ours failed ***"
  run ref_s$S  SEED=$S CERT_LAMBDA=0.3 || echo "*** seed $S: reference failed ***"
}
for S in 1 2 3; do chain $S & done
wait
echo "=== SCALE-FT-001 complete ==="
