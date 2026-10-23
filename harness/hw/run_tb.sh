#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
~/research/venv_gpu/bin/python tb_core.py ~/research/data/e6 "$@"
