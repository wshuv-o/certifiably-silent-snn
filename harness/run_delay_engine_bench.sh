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
echo "=== waiting for an idle CPU ==="
while pgrep -f "s5_delays.py|s4_improve.py" > /dev/null; do sleep 30; done
echo "=== CPU idle at $(date +%H:%M:%S); nproc=$(nproc) ==="
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
