#!/usr/bin/env bash
# ENGINE-002b: dense 512-ALIF models as 8 cores x 64 neurons (fits 12 hardware threads).
set -e
cd "$(dirname "$0")"
source ~/research/venv_gpu/bin/activate
CORES=8 python -u s2_export.py 2>&1 | grep -vE "Warning|warn\("
for tag in control certified; do ~/research/build/s2_engine ~/research/data/e2 $tag; done