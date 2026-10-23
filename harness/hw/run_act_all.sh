#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
# HW-005 batch: data-driven power of the neuron unit and pipelined certificate units, two networks
cd $REPO/harness/hw
for m in lk4 hdcls; do
  for cfg in "nu 1" "pcu2 2" "pcu4 4"; do
    set -- $cfg
    ~/research/venv_gpu/bin/python activity.py $1 $2 $HOME/research/data/e6 $m 5000 2>&1 | grep RESULT
  done
done
echo ACT_DONE
