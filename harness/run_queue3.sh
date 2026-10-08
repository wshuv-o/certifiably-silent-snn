#!/usr/bin/env bash
# Single queue, so two queues cannot both see a free slot and overshoot the three-process cap.
#
# Priority. The accuracy question is open and the recipe attribution answers it; the learnable-delay
# replication confirms an effect that is already a fiftyfold gap on seed 1 and will not reverse, so it
# yields less per GPU-hour right now. Seed 2 is kept so table 1's corrected rows rest on more than one
# run; seed 3 is left for an idle window.
set -u
cd "$(dirname "$0")"
LOG=~/research/logs

running() { pgrep -fc "python -u s[57]_" 2>/dev/null || echo 0; }

launch() {
  local name=$1; shift
  while [ "$(running)" -ge 3 ]; do sleep 30; done
  echo "$(date +%H:%M:%S) launching $name ($(running) running)"
  setsid nohup "$@" > "$LOG/$name.log" 2>&1 < /dev/null &
  sleep 45
}

echo "=== queue3 start $(date) ==="
launch q_nobins env ARM=nobins bash run_acc_ladder.sh
launch q_nodrop env ARM=nodrop bash run_acc_ladder.sh
launch q_dcls2  env S=2        bash run_dcls002_seeds.sh
launch q_nobn   env ARM=nobn   bash run_acc_ladder.sh
launch q_nooc   env ARM=nooc   bash run_acc_ladder.sh
launch q_atan   env ARM=atan   bash run_acc_ladder.sh
launch q_dcls3  env S=3        bash run_dcls002_seeds.sh
echo "=== queue3 submitted $(date) ==="
