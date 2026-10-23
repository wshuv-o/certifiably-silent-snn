#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
LP=1 ~/research/venv_gpu/bin/python tb_core.py $HOME/research/data/e6 "$@"
