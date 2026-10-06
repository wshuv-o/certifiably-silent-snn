#!/usr/bin/env bash
# FINAL speed (idle CPU): ExCap vs control, small (8x32 local, seeds 1-3) and strong (512-ALIF dense 8x64, seeds 2-4).
set -e
cd "$(dirname "$0")"
M=~/research/models
source ~/research/venv_speech/bin/activate
L=""
for s in 1 2 3; do L="$L,sm_ctrl_s$s=lif_local:$M/p005_s${s}_l0.pt,sm_excap_s$s=lif_local:$M/alt_clamp_s$s.pt"; done
for s in 2 3 4; do L="$L,dn_ctrl_s$s=alif:$M/conf_ctrl_s$s.pt,dn_excap_s$s=alif:$M/conf_excap_s$s.pt"; done
CORES=8 MODELS="${L#,}" python -u s3_export.py 2>&1 | grep -vE "Warning|warn\("
g++ -O3 -march=native -std=c++20 -pthread s3_engine.cpp -o ~/research/build/s3_engine
for s in 1 2 3; do for k in ctrl excap; do ~/research/build/s3_engine ~/research/data/e3 sm_${k}_s$s 2; done; done
for s in 2 3 4; do for k in ctrl excap; do ~/research/build/s3_engine ~/research/data/e3 dn_${k}_s$s 2; done; done
