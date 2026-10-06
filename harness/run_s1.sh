#!/usr/bin/env bash
# S1 engine test (pre-registered in research/N3_SCALEUP_PLAN.md)
source ~/research/venv_speech/bin/activate
cd "$(dirname "$0")"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u s1_engine.py 2>&1 | grep --line-buffered -vE "Warning|warn\("
