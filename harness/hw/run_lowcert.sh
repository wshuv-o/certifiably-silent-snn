#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
~/research/venv_gpu/bin/python lowcert.py check $HOME/research/data/e6 lk4 8000
~/research/venv_gpu/bin/python lowcert.py check $HOME/research/data/e6 hdcls 8000
