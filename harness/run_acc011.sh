#!/usr/bin/env bash
# ACC-011 (pre-registered): SpikCommander under our protocol. Seed given as $1.
source ~/research/venv_gpu/bin/activate
export PYTHONPATH=~/research/refs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/research/refs/SCommander-protocol
s=$1; f=~/research/logs/a11_s$s.log
grep -qas "^RESULT" "$f" && exit 0
echo "--- a11 s$s $(date +%H:%M:%S) ---"; SEED=$s TAG=a11 TEST=1 python -u scmd_main.py > "$f" 2>&1
grep -a "^RESULT" "$f" || { echo "*** FAILED a11 s$s ***"; tail -15 "$f"; }
