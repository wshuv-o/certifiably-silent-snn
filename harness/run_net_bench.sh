#!/usr/bin/env bash
# NET-ENGINE-001 benchmark: REAL TCP, no emulated latency. Two ranks, cores split between them.
#
# Realistic deployment shape: each rank is a "chip" holding many cores. Within a rank, cores share
# memory (free); across ranks every spike and certificate crosses a real socket. That cross-rank
# dependency is exactly what certificates are supposed to relieve.
#
# TIMING HYGIENE: waits for an explicit all-clear plus sustained idle, after two earlier CPU timings
# were spoiled by concurrent jobs (once by a race in the idle check itself).
#
# TWO-MACHINE RUN (removes loopback from the picture entirely):
#   on host A:  s8_engine_net ~/research/data/e6 <tag> <mode> 0 2 58300
#   on host B:  s8_engine_net ~/research/data/e6 <tag> <mode> 1 2 58300 <ip-of-A>
# Both hosts need the same exported files in ~/research/data/e6.
set -e
cd "$(dirname "$0")"
source ~/research/venv_gpu/bin/activate
B=~/research/build; M=~/research/models
echo "=== waiting for all GPU/training work to finish ==="
while pgrep -f "s5_delays.py|s7_dcls.py|s4_improve.py" > /dev/null; do sleep 30; done
idle=0; while [ "$idle" -lt 5 ]; do
  if pgrep -f "s5_delays.py|s7_dcls.py|s4_improve.py" > /dev/null; then idle=0; else idle=$((idle+1)); fi
  sleep 20
done
echo "=== sustained-idle at $(date +%H:%M:%S), nproc=$(nproc) ==="
g++ -O3 -march=native -std=c++20 -pthread s8_engine_net.cpp -o "$B/s8_engine_net"
PORT=58300
for CORES in 8 16 32; do
  echo "########## CORES=$CORES (2 ranks, $((CORES/2)) cores each) $(date +%H:%M:%S) ##########"
  CORES=$CORES NSAMP=100 DELAYS=2,4,8 \
    MODELS="nb_cert=$M/d3_F2_cert10.pt,nb_ctrl=$M/d2_E3_mindelay2_long.pt" \
    python -u s6_export_delays.py 2>&1 | grep -vE "Warning|warn\(" || true
  for tag in nb_cert nb_ctrl; do
    for MODE in 0 1; do
      PORT=$((PORT+1))
      "$B/s8_engine_net" ~/research/data/e6 $tag $MODE 0 2 $PORT > /tmp/nb0.txt 2>&1 &
      sleep 1
      "$B/s8_engine_net" ~/research/data/e6 $tag $MODE 1 2 $PORT 127.0.0.1 > /tmp/nb1.txt 2>&1 || true
      wait || true
      cat /tmp/nb0.txt
    done
  done
done
echo "=== NET BENCH COMPLETE $(date +%H:%M:%S) ==="
