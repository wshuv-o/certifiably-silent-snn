#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
NOREPAIR=1 OUTSUFFIX= ~/research/venv_gpu/bin/python pnr.py nu 1 24 > /dev/null 2>&1
L=build/pnr_nu/flow.log
echo "RESULT nu $(grep -m1 UPSIZED $L) $(grep -m1 'Design area' $L) | $(grep -m1 'worst slack' $L) | slewviol=$(grep -c '(VIOLATED)' $L) | drc=$(grep -c -i violation build/pnr_nu/drc.rpt) | errors=$(grep -c -E '^\[ERROR|^Error' $L)"
~/research/venv_gpu/bin/python activity.py nu 1 $HOME/research/data/e6 lk4 5000 | grep RESULT
python3 - <<'PY'
import re, statistics
t = open("build/pnr_nu/power_data.tcl").read()
acts = [float(a) for a in re.findall(r"-pins \{\S+\} -activity ([\d.]+)", t)]
print("NU_MEAN_DENSITY %.4f" % statistics.mean([a for a in acts if a < 1.5]))
PY
echo NU_DONE
