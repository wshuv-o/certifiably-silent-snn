#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
for d in lcu4 nu; do
  K=4; [ $d = nu ] && K=1
  NOREPAIR=1 ~/research/venv_gpu/bin/python pnr.py $d $K 24 > /dev/null 2>&1
  L=build/pnr_$d/flow.log
  echo "RESULT $d $(grep -m1 UPSIZED $L) $(grep -m1 'Design area' $L) | $(grep -m1 'worst slack' $L) | slewviol=$(grep -c '(VIOLATED)' $L) | drc=$(grep -c -i violation build/pnr_$d/drc.rpt) | errors=$(grep -c -E '^\[ERROR|^Error' $L)"
done
~/research/venv_gpu/bin/python act_low.py $HOME/research/data/e6 lk4 5000
~/research/venv_gpu/bin/python act_low.py $HOME/research/data/e6 hdcls 5000
~/research/venv_gpu/bin/python activity.py nu 1 $HOME/research/data/e6 lk4 5000 | grep RESULT
python3 - <<'PY'
import re, statistics
t = open("build/pnr_nu/power_data.tcl").read()
acts = [float(a) for a in re.findall(r"-pins \{\S+\} -activity ([\d.]+)", t)]
m = statistics.mean([a for a in acts if a < 1.5]); print("NU_MEAN_DENSITY %.4f" % m)
head = t.split("set_power_activity -input")[0]
open("build/pnr_nu/pmean.tcl", "w").write(head + f"set_power_activity -global -activity {m:.4f}\nset_power_activity -input -activity {m:.4f}\nset_power_activity -input_ports clk -activity 2\nreport_power\nexit\n")
PY
(cd build && ~/eda/bin/openroad -no_init -exit pnr_nu/pmean.tcl 2>&1 | grep -E "^Total " | sed 's/^/NU_PMEAN /')
echo SEQ_DONE
