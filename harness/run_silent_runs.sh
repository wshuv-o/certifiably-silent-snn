#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness
for m in cert lk4 lk8 hdcls hscmd; do nice -n 19 ~/research/venv_gpu/bin/python silent_runs.py $HOME/research/data/e6 $m 50; done
