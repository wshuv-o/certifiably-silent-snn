#!/usr/bin/env bash
# SSC-FROZEN-001 (pre-registered in N3_SCALEUP_PLAN.md): the SHD-winning recipe, FROZEN, on SSC.
# No retuning. Epochs matched by GRADIENT STEPS (SHD 150 ep x 55 = 8250 steps; SSC 578 steps/ep -> 14 ep).
# Test set evaluated once, for both arms.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/sf_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/sf_${tag}.pt python -u s5_delays.py > "$LOG/sf_$tag.log" 2>&1
    grep -qas "^RESULT" "$LOG/sf_$tag.log" && { echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0; }
    echo "retry $tag attempt $a: $(grep -aoE 'Error[A-Za-z]*|CUDA error: [a-z ]+' "$LOG/sf_$tag.log" | head -1)"; sleep 5
  done
  echo "*** FAILED: $tag ***"; tail -15 "$LOG/sf_$tag.log"; return 1
}
FROZEN="DATASET=ssc H=512 SEED=1 DELAYS=2,4,8 AUG=2 TAUM=2.0 TEST=1"
echo "=== SSC-FROZEN-001 START $(date +%H:%M:%S) ==="
runone S1_ssc_ctrl $FROZEN EPOCHS=14 || exit 1
runone S2_ssc_cert $FROZEN EPOCHS=14 INIT=$M/sf_S1_ssc_ctrl.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0
echo "=== SSC-FROZEN-001 COMPLETE $(date +%H:%M:%S) ==="
