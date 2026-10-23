#!/usr/bin/env bash
# RECERT-001 (pre-registered in N3_SCALEUP_PLAN.md): answer two reviewer objections by RE-CERTIFYING
# existing trained checkpoints, with no retraining. EPOCHS=0 loads a checkpoint and goes straight to
# evaluation, so this is evaluation cost only.
#   (a) "Why K=4?"  -> sweep the certificate horizon K in {2,4,8}.
#   (b) "Your cores are far smaller than real neuromorphic cores" -> sweep neurons-per-core CPC in
#       {32,64,128}. Certification needs a WHOLE core silent, so larger cores must certify less often;
#       the question is how fast it falls, since Loihi-class cores hold ~1k neurons.
# The trained weights are identical throughout; only the certificate's horizon and the core partition
# change. Any drop is therefore a property of the certificate, not of a different network.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$LOG"
run() { tag=$1; shift
  if grep -qas "^RESULT" "$LOG/rc_$tag.log"; then echo "SKIP $tag"; return 0; fi
  env TAG=$tag EPOCHS=0 "$@" python -u "$SCRIPT" > "$LOG/rc_$tag.log" 2>&1
  grep -qas "^RESULT" "$LOG/rc_$tag.log" && echo "OK $tag" || { echo "*** FAILED: $tag ***"; tail -6 "$LOG/rc_$tag.log"; }
}
echo "=== RECERT-001 START $(date +%H:%M:%S) ==="
# --- fixed-tap model (the SHD winner) ---
SCRIPT=s5_delays.py
for K in 2 4 8; do for CPC in 32 64 128; do
  run f_K${K}_C${CPC} H=512 SEED=1 DELAYS=2,4,8 AUG=2 K=$K CPC=$CPC INIT=$M/d3_F2_cert10.pt
done; done
# --- learnable-delay model (certifiable without a penalty) ---
SCRIPT=s7_dcls.py
for K in 2 4 8; do for CPC in 32 64 128; do
  run d_K${K}_C${CPC} H=512 SEED=1 DMIN=2 DMAX=8 AUG=2 K=$K CPC=$CPC INIT=$M/dc_DC1_ctrl.pt
done; done
echo "=== RECERT-001 COMPLETE $(date +%H:%M:%S) ==="
