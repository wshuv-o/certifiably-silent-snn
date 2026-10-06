#!/usr/bin/env bash
# B: general engine with distributed recursive certificates on existing models.
set -e
cd "$(dirname "$0")"
M=~/research/models
source ~/research/venv_speech/bin/activate
CORES=8 MODELS="small_cert=lif_local:$M/p005_s1_l0.1.pt,small_clamp=lif_local:$M/alt_clamp_s1.pt,small_l1=lif_local:$M/alt_l1_s1.pt,dense_ctrl=alif:$M/s2_l0_s1.pt,dense_cert=alif:$M/s2_l0.3_s1.pt" \
  python -u s3_export.py 2>&1 | grep -vE "Warning|warn\("
g++ -O3 -march=native -std=c++20 -pthread s3_engine.cpp -o ~/research/build/s3_engine
for tag in small_cert small_clamp small_l1 dense_ctrl dense_cert; do ~/research/build/s3_engine ~/research/data/e3 $tag; done