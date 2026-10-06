#!/usr/bin/env bash
# Watcher: when SSC lambda=0 finishes, kill the q2-launched lambda=0.3 (it would exceed q2's 2 h limit)
# and run lambda=0.3 in this process with its own time budget.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
until [ -f ../results/s2_ssc_none_l0_s1.json ]; do sleep 20; done
sleep 15
for pid in $(pgrep -f "s2_strong.py"); do
  if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q '^DATASET=ssc$' && \
     tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q '^CERT_LAMBDA=0.3$'; then
    echo "killing q2 duplicate pid $pid"; kill $pid
  fi
done
sleep 5
echo "=== SSC lambda=0.3 (own process)"
DATASET=ssc EPOCHS=15 SEED=1 CERT_LAMBDA=0.3 SAVE=~/research/models/ssc_l0.3_s1.pt \
  python -u s2_strong.py 2>&1 | grep --line-buffered -E "^(RESULT|epoch)"