#!/usr/bin/env bash
# EXP-006: workload-independent memory primitives, then pre-registered prediction check.
source ~/research/venv/bin/activate
cd "$(dirname "$0")"
timeout 3000 python exp006_micro.py > /tmp/exp006_micro.log 2>&1
echo "micro exit=$?"
grep -E '^(cpu|gpu) ' /tmp/exp006_micro.log
python exp006_predict.py
