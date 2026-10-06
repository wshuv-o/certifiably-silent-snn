#!/usr/bin/env bash
# CORE-SCALING-001 (pre-registered in N3_SCALEUP_PLAN.md): exact-execution speed-up vs CORE COUNT.
# The manuscript reports 8 cores measured on a 6-core i7 (oversubscribed). This machine has 32 threads,
# so 2..32 cores can be measured without oversubscription for the first time.
# Waits for GPU training to finish first: CPU wall-clock timing requires an idle machine.
set -e
cd "$(dirname "$0")"
source ~/research/venv_gpu/bin/activate
M=~/research/models; B=~/research/build; mkdir -p "$B"
echo "=== waiting for GPU jobs to finish (CPU must be idle for timing) ==="
while pgrep -f "s4_improve.py" > /dev/null; do sleep 20; done
echo "=== CPU idle at $(date +%H:%M:%S); compiling engine ==="
g++ -O3 -march=native -std=c++20 -pthread s3_engine.cpp -o "$B/s3_engine"
nproc
# Confirmed-recipe models at H=512 (CONFIRM-002 winner) + their controls. seed 2.
for CORES in 2 4 8 16 32; do
  echo "########## CORES=$CORES  $(date +%H:%M:%S) ##########"
  CORES=$CORES MODELS="sc_ours=alif:$M/c2_ours_s2.pt,sc_ctrl=alif:$M/c2_ctrl_s2.pt" \
    python -u s3_export.py 2>&1 | grep -vE "Warning|warn\(" || true
  for m in sc_ours sc_ctrl; do
    echo "---- $m at CORES=$CORES ----"
    "$B/s3_engine" ~/research/data/e3 $m 2
  done
done
echo "=== CORE-SCALING-001 COMPLETE $(date +%H:%M:%S) ==="
