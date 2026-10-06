#!/usr/bin/env bash
# BUDGET-001 (pre-registered in N3_SCALEUP_PLAN.md): do a faster membrane leak (TAUM) and/or ring-local
# connectivity (LOCAL) cut the certification cost by changing R/budget instead of crushing weights?
# Each config gets its OWN matched control. SERIAL (WSL GPU concurrency unreliable). Validation only.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
run() { tag=$1; shift
  echo "--- $tag $(date +%H:%M:%S) ---"
  env TAG=$tag "$@" SAVE=$M/b1_${tag}.pt python -u s4_improve.py > "$LOG/b1_$tag.log" 2>&1
  grep -q "^RESULT" "$LOG/b1_$tag.log" && echo "OK $tag" || { echo "*** FAILED: $tag ***"; tail -8 "$LOG/b1_$tag.log"; }
}
cfg() {  # id TAUM LOCAL
  id=$1; tm=$2; lc=$3
  run ${id}_ctrl SEED=1 TAUM=$tm LOCAL=$lc
  run ${id}_ours SEED=1 TAUM=$tm LOCAL=$lc INIT=$M/b1_${id}_ctrl.pt EPOCHS=40 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3
}
echo "=== BUDGET-001 START $(date +%H:%M:%S) ==="
cfg B0 2.0 0
cfg B1 1.0 0
cfg B2 0.5 0
cfg B3 2.0 1
cfg B4 0.5 1
echo "=== BUDGET-001 COMPLETE $(date +%H:%M:%S) ==="
