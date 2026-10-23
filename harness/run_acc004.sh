#!/usr/bin/env bash
# ACC-004 (pre-registered in research/N3_SCALEUP_PLAN.md): published DCLS SHD code, speaker-disjoint
# validation, test accuracy at the best-validation epoch. Seeds 0, 1, 2 in sequence.
source ~/research/venv_gpu/bin/activate
export PYTHONPATH=~/research/refs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=~/research/logs; mkdir -p "$LOG" ~/research/models
python "$(dirname "$0")/make_ref_protocol.py" || exit 1
cd ~/research/refs/SNN-delays-protocol
for s in 0 1 2; do
  f=$LOG/acc004_s$s.log
  grep -qs "^PROTOCOL_RESULT" "$f" && { echo "SKIP s$s"; continue; }
  echo "--- seed $s $(date +%H:%M:%S) ---"
  SEED=$s python -u main.py > "$f" 2>&1
  r=$(grep -a "^PROTOCOL_RESULT" "$f")
  [ -n "$r" ] || { echo "*** FAILED s$s ***"; tail -20 "$f"; continue; }
  best_test=$(grep -a "Acc Test" "$f" | sed 's/.*Best Acc Test = \([0-9.]*\)%.*/\1/' | tail -1)
  echo "$r test_selected_max=$best_test"
done
echo "=== ACC-004 END $(date) ==="
