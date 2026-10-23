#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
for m in lk4 cert hdcls hscmd; do ~/research/venv_gpu/bin/python early_abort.py $HOME/research/data/e6 $m 3; done
