#!/usr/bin/env bash
# DELAY-ENGINE-001 benchmark (pre-registered in N3_SCALEUP_PLAN.md).
# Exact-execution speed-up of certificates on the DELAY architecture, vs a handshake baseline that is
# GIVEN the free lookahead the minimum delay provides (d_min=2 => a core needs neighbour spikes only up
# to t+1-2). Anything less would be a strawman: that lookahead belongs to the delays, not to us.
#
# TIMING HYGIENE: blocks until no training process remains. CPU wall-clock measurement requires an idle
# machine, and thread oversubscription has already invalidated one engine run in this project.
set -e
cd "$(dirname "$0")"
source ~/research/venv_gpu/bin/activate
B=~/research/build; M=~/research/models; mkdir -p "$B"
# GATING (fixed 2026-10-07 after a race): the previous version polled "is any training running?" and
# happened to sample the one-second gap between two sequential training runs, so it benchmarked for 14
# minutes alongside a live job and produced invalid timings for the second time. Now it requires BOTH
# an explicit completion marker from the orchestrator AND a sustained idle period, so a momentary gap
# between queued runs cannot be mistaken for an idle machine.
echo "=== waiting for the orchestrator to signal all GPU work complete ==="
while ! grep -aq "OVERNIGHT GPU WORK COMPLETE" ~/research/logs/overnight.log 2>/dev/null; do sleep 60; done
echo "=== marker seen; requiring 5 consecutive idle checks ==="
idle=0
while [ "$idle" -lt 5 ]; do
  if pgrep -f "s5_delays.py|s4_improve.py" > /dev/null; then idle=0; else idle=$((idle+1)); fi
  sleep 20
done
echo "=== CPU sustained-idle at $(date +%H:%M:%S); nproc=$(nproc) ==="
g++ -O3 -march=native -std=c++20 -pthread s6_engine_delays.cpp -o "$B/s6_engine_delays"
for CORES in 4 8 16 32; do
  echo "########## CORES=$CORES  $(date +%H:%M:%S) ##########"
  CORES=$CORES NSAMP=200 DELAYS=2,4,8 \
    MODELS="dl_cert=$M/d3_F2_cert10.pt,dl_ctrl=$M/d2_E3_mindelay2_long.pt" \
    python -u s6_export_delays.py 2>&1 | grep -vE "Warning|warn\(" || true
  for m in dl_cert dl_ctrl; do
    echo "---- $m at CORES=$CORES ----"
    "$B/s6_engine_delays" ~/research/data/e6 $m 2
  done
done
echo "=== DELAY-ENGINE BENCH COMPLETE $(date +%H:%M:%S) ==="
