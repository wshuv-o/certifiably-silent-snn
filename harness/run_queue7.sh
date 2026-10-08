#!/usr/bin/env bash
# ACC-002b, the last accuracy arm, after the speed diagnostic has had the idle machine.
set -u
cd "$(dirname "$0")"
LOG=~/research/logs
while ! grep -qa "SPEED-DIAG-001 COMPLETE\|FATAL" "$LOG/speed_diag.log" 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) diagnostic finished; launching ACC-002b"
running() { pgrep -fc "python -u s[57]_" 2>/dev/null; true; }
while [ "$(running)" -ge 2 ]; do sleep 30; done
setsid nohup env TAG=ac_t_ctrl100 H=512 SEED=1 AUG=2 DELAYS=2,4,8 EPOCHS=150 \
  TAU_LEARN=1 LR_TAU_MULT=100 SAVE=$HOME/research/models/ac_t_ctrl100.pt \
  bash -c "source ~/research/venv_gpu/bin/activate && python -u s5_delays.py" \
  > "$LOG/ac_t_ctrl100.log" 2>&1 < /dev/null &
echo "launched"
