#!/usr/bin/env bash
# Run the remaining pre-registered experiments back to back, unattended.
#
# Serial by measurement: with three concurrent processes each run slowed from 4 s/epoch to
# 10-15 s/epoch, so aggregate throughput was a wash while the critical path grew. The workload is
# kernel-launch bound, so a GPU sitting at 25% is not idle capacity we can use.
#
# Ordered to front-load information. Per-epoch cost goes as NIN*H + |D|*H^2, so against a measured
# 11 min at H=512 the H=1024 runs cost ~3.4x and the H=2048 runs ~12.3x. Running the two cheap
# widths and the cross-modality check before the H=2048 pair yields three of the four answers by
# midnight instead of one, at the same finishing time.
#
# Every stage skips work whose log already holds a RESULT line, so this survives a relaunch.
cd "$(dirname "$0")"
LOG=~/research/logs

ndone() { grep -ahcE "^RESULT" $LOG/g2_*.log 2>/dev/null | paste -sd+ | bc; }

echo "=== QUEUE START $(date) ==="

# 1. GOVERN-002 is already in flight under its own driver
while [ "$(ndone)" -lt 13 ]; do sleep 60; done
echo "--- GOVERN-002 complete ($(ndone)/13) $(date) ---"

# 2. width scaling at the two cheap widths: first signal on whether bounded fan-in holds
# Measured, not scaled from FLOPs: the step is launch-bound, so H=2048 costs 160 ms against
# H=512's 106 ms -- 1.5x, not the 12x a FLOP count suggests. The whole width sweep is ~94 min,
# so it runs as one block and completes an hour earlier than when it was split.
./run_scalelocal001.sh A
./run_scalelocal001.sh B
./run_scalelocal001.sh C
echo "--- SCALE-LOCAL-001 complete $(date) ---"

# 3. cross-modality check
./run_nmnist001.sh
echo "=== QUEUE COMPLETE $(date) ==="
