#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
# HW-004 runs: neuron unit (registered) and the pipelined certificate units, routed in sky130
cd $REPO/harness/hw
P=~/research/venv_gpu/bin/python
for cfg in "nu 1 24 0" "pcu4 4 24 1" "pcu2 2 24 1"; do
  set -- $cfg
  if [ "$4" = 1 ]; then PIPE=1 $P pnr.py $1 $2 $3 > /dev/null 2>&1; else $P pnr.py $1 $2 $3 > /dev/null 2>&1; fi
  L=build/pnr_$1/flow.log
  echo "RESULT $1 K=$2 clk_ns=$3 $(grep -m1 'Design area' $L) | $(grep -m1 'worst slack' $L) | $(grep -m1 '^Total ' $L) | drc=$(grep -c -i violation build/pnr_$1/drc.rpt)"
done
echo PNR_DONE
