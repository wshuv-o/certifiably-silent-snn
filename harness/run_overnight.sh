#!/usr/bin/env bash
# Overnight orchestrator. Runs the remaining pre-registered work STRICTLY IN SEQUENCE so that no more
# than 3 concurrent CUDA processes ever exist (6 caused CUDA_ERROR_UNKNOWN and destroyed 9 runs; even
# 4 is above the level this WSL/Blackwell setup has been shown to tolerate).
# Order is by scientific value: the test-set confirmation of the frozen recipe comes first, because
# without it none of tonight's numbers belong in a manuscript.
cd "$(dirname "$0")"
echo "=== OVERNIGHT START $(date +%H:%M:%S) ==="
echo "--- waiting for SSC to finish ---"
while pgrep -f "s5_delays.py" > /dev/null; do sleep 30; done
echo "=== [1/3] SHD-CONFIRM-FROZEN (test set, seeds 2-4) $(date +%H:%M:%S) ==="
bash run_shd_confirm_frozen.sh || echo "*** SHD-CONFIRM had failures (see log) ***"
echo "=== [2/3] DELAY-004 (300-epoch runs) $(date +%H:%M:%S) ==="
bash run_delay004.sh || echo "*** DELAY-004 had failures (see log) ***"
echo "=== [3/3] engine benchmark runs itself once the CPU is idle $(date +%H:%M:%S) ==="
echo "=== OVERNIGHT GPU WORK COMPLETE $(date +%H:%M:%S) ==="
