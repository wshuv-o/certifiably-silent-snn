#!/usr/bin/env bash
# Waits for queue3 to finish submitting before taking any slot, so two schedulers never both observe
# a free one and overshoot the three-process cap this machine enforces by faulting.
set -u
cd "$(dirname "$0")"
LOG=~/research/logs
while pgrep -f "run_queue3.s[h]" > /dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) queue3 done submitting; queue4 starting"
running() { pgrep -fc "python -u s[57]_" 2>/dev/null; true; }
launch() {
  local name=$1; shift
  while [ "$(running)" -ge 3 ]; do sleep 30; done
  echo "$(date +%H:%M:%S) launching $name ($(running) running)"
  setsid nohup "$@" > "$LOG/$name.log" 2>&1 < /dev/null &
  sleep 45
}
launch q_acc002 bash run_acc002.sh
echo "=== queue4 submitted $(date) ==="
