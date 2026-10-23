#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
while pgrep -f "pnr.py nu" > /dev/null; do sleep 5; done
NOREPAIR=1 ~/research/venv_gpu/bin/python pnr.py lcu4 4 24 > /dev/null 2>&1
L=build/pnr_lcu4/flow.log
echo "RESULT lcu4 $(grep -m1 UPSIZED $L) $(grep -m1 'Design area' $L) | $(grep -m1 'worst slack' $L) | slewviol=$(grep -c '(VIOLATED)' $L) | drc=$(grep -c -i violation build/pnr_lcu4/drc.rpt) | errors=$(grep -c -E '^\[ERROR|^Error' $L)"
~/research/venv_gpu/bin/python act_low.py $HOME/research/data/e6 lk4 5000
~/research/venv_gpu/bin/python act_low.py $HOME/research/data/e6 hdcls 5000
echo LCU_DONE
