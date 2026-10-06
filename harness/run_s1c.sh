#!/usr/bin/env bash
# S1c: final pre-registered speed attempt (Lemma-1 K-step certificate). Uses S1b exports.
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -std=c++20 -pthread s1b_engine.cpp -o ~/research/build/s1b_engine
for tag in control certified; do
  ~/research/build/s1b_engine ~/research/data/s1b $tag 1
done
