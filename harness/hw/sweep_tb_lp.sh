#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
# HW-001: co-simulate every core of each network on NS samples; any mismatch aborts that run (exit 1).
cd $REPO/harness/hw
NS=${NS:-3}
mkdir -p ../../results/hw008
for m in lk4 cert hdcls hscmd; do
  for p in $(seq 0 15); do
    LP=1 ~/research/venv_gpu/bin/python tb_core.py $HOME/research/data/e6 $m $p $NS > ../../results/hw008/${m}_core${p}.log 2>&1 || echo "FAIL $m $p"
    tail -1 ../../results/hw008/${m}_core${p}.log
  done
done
echo SWEEP_DONE
