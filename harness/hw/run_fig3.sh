#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw
~/research/venv_gpu/bin/python fig_hw.py build ~/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef build/fig_hw.pdf
