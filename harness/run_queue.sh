#!/usr/bin/env bash
# Run the remaining pre-registered experiments back to back, unattended.
#
# Serial by measurement, not by superstition: with three concurrent processes each run slowed from
# 4 s/epoch to 10-15 s/epoch, so aggregate throughput was a wash while the critical path grew. The
# workload is kernel-launch bound, so the GPU sitting at 25% is not idle capacity we can use.
#
# Every stage skips work whose log already holds a RESULT line, so this can be relaunched after a
# WSL restart without losing anything.
cd "$(dirname "$0")"
LOG=~/research/logs

ndone() { grep -ahcE "^RESULT" $LOG/g2_*.log 2>/dev/null | paste -sd+ | bc; }

echo "=== QUEUE START $(date) ==="

# 1. let GOVERN-002 finish; its three chains are already in flight
while [ "$(ndone)" -lt 13 ]; do sleep 60; done
echo "--- GOVERN-002 complete ($(ndone)/13) $(date) ---"

# 2. width scaling under bounded fan-in, serial: H=2048 is ~10x the cost of H=512 per epoch
./run_scalelocal001.sh A
./run_scalelocal001.sh B
./run_scalelocal001.sh C
echo "--- SCALE-LOCAL-001 complete $(date) ---"

# 3. cross-modality check
./run_nmnist001.sh
echo "=== QUEUE COMPLETE $(date) ==="
