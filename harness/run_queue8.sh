#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
while [ "$(pgrep -fc "python -u s[57]_" 2>/dev/null; true)" -ge 1 ]; do sleep 30; done
echo "$(date +%H:%M:%S) launching seed-3 replication"
setsid nohup env S=3 bash run_dcls002_seeds.sh > ~/research/logs/q_dcls3.log 2>&1 < /dev/null &
