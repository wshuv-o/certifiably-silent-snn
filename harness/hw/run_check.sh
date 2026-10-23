#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
for m in "$@"; do ~/research/venv_gpu/bin/python check_golden.py ~/research/data/e6 "$m" 5; done
