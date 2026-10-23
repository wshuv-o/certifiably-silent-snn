#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
# HW-004/005 rerun with design repair: route the three units, then measured-activity power on two networks
cd $REPO/harness/hw
bash run_pnr.sh
bash power.sh > /dev/null 2>&1
for m in lk4 hdcls; do
  for cfg in "nu 1" "pcu2 2" "pcu4 4"; do
    set -- $cfg
    ~/research/venv_gpu/bin/python activity.py $1 $2 $HOME/research/data/e6 $m 5000 2>&1 | grep RESULT
  done
done
for d in nu pcu2 pcu4; do echo "SLEWVIOL $d $(grep -c '(VIOLATED)' build/pnr_$d/flow.log)"; done
echo HWP_DONE
