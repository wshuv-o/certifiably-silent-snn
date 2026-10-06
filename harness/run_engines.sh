#!/usr/bin/env bash
# ENGINE-ALT + ENGINE-002. Run when the CPU is free of training jobs.
set -e
cd "$(dirname "$0")"
M=~/research/models
source ~/research/venv_speech/bin/activate
MODELS="clamp=$M/alt_clamp_s1.pt,l1=$M/alt_l1_s1.pt" python -u s1b_export.py 2>&1 | grep -vE "Warning|warn\("
g++ -O3 -march=native -std=c++20 -pthread s1b_engine.cpp -o ~/research/build/s1b_engine
for tag in clamp l1; do ~/research/build/s1b_engine ~/research/data/s1b $tag 1; done
deactivate
source ~/research/venv_gpu/bin/activate
python -u s2_export.py 2>&1 | grep -vE "Warning|warn\("
g++ -O3 -march=native -std=c++20 -pthread s2_engine.cpp -o ~/research/build/s2_engine
for tag in control certified; do ~/research/build/s2_engine ~/research/data/e2 $tag; done