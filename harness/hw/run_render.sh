#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw/build
~/research/venv_gpu/bin/python ../render_gds.py pnr_lcu4/lcu4.def gds_lcu4 25 2>&1 | tail -5
