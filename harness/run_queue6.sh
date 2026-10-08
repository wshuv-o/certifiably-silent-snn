#!/usr/bin/env bash
# Run the saturation diagnostic the moment the machine is genuinely idle, then the seed-3 replication.
#
# Conditions on the DRIVER scripts being gone, not just on zero training processes: run_acc002.sh
# runs two trainings back to back and there is a sub-second gap between them during which a
# process-count test would wrongly report an idle machine.
set -u
cd "$(dirname "$0")"
LOG=~/research/logs
busy() {
  pgrep -f "run_acc002.s[h]|run_dcls002.s[h]|run_acc_ladder.s[h]|run_queue4.s[h]" > /dev/null && return 0
  [ "$(pgrep -fc "python -u s[57]_" 2>/dev/null || echo 0)" -gt 0 ] && return 0
  return 1
}
while busy; do sleep 60; done
sleep 60                      # let the machine settle before timing anything
echo "$(date +%H:%M:%S) machine idle; running SPEED-DIAG-001"
bash run_speed_diag.sh > "$LOG/speed_diag.log" 2>&1
echo "$(date +%H:%M:%S) diagnostic done rc=$?; starting seed-3 replication"
setsid nohup env S=3 bash run_dcls002_seeds.sh > "$LOG/q_dcls3.log" 2>&1 < /dev/null &
