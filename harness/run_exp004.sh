#!/usr/bin/env bash
# EXP-004: working-set sweep (H-cache). Run inside WSL from harness/.
# Usage: run_exp004.sh [log2N ...]   (default: 17..22)
source ~/research/venv/bin/activate
cd "$(dirname "$0")"
for k in ${@:-17 18 19 20 21 22}; do
  python bench_sparsity.py --N $((1 << k)) --F 32 --windows 0 --seed 4 \
      --out ../results/exp004_n$k > /tmp/exp004_n$k.log 2>&1
  echo "n$k: exit=$? $(grep -E 'VERIFICATION' /tmp/exp004_n$k.log) $(tail -1 /tmp/exp004_n$k.log | grep -iE 'error|killed')"
done
python analyze_cache.py
