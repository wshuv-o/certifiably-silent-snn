#!/usr/bin/env bash
# S1b: export models, build C++ engine, run handshake vs cert for control and certified models.
set -e
cd "$(dirname "$0")"
source ~/research/venv_speech/bin/activate
python -u s1b_export.py 2>&1 | grep -vE "Warning|warn\("
mkdir -p ~/research/build
g++ -O3 -march=native -std=c++20 -pthread s1b_engine.cpp -o ~/research/build/s1b_engine
for tag in control certified; do
  ~/research/build/s1b_engine ~/research/data/s1b $tag
done
