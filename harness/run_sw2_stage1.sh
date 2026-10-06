#!/usr/bin/env bash
# SWEEP-002 stage 1: S0 control (teacher/init) and S1 scratch certificate loss. Seed 1.
# Logging note (2026-10-06): output goes to ~/research/logs/<tag>.log instead of being piped
# through grep. The old `python ... | grep -E "^(RESULT|...)"` form discarded tracebacks, which
# silently hid a failed torch.save and left stage 2 with no INIT/TEACHER checkpoint.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
( env SEED=1 TAG=S0_ctrl   SAVE=$M/sw2_S0_ctrl_s1.pt   python -u s4_improve.py > "$LOG/S0_ctrl.log" 2>&1 ) &
( env SEED=1 TAG=S1_cert03 CERT_LAMBDA=0.3 SAVE=$M/sw2_S1_cert03_s1.pt python -u s4_improve.py > "$LOG/S1_cert03.log" 2>&1 ) &
wait
fail=0
for t in S0_ctrl S1_cert03; do
  if grep -q "^RESULT" "$LOG/$t.log"; then grep -h "^RESULT" "$LOG/$t.log"; else
    echo "*** FAILED: $t ***"; tail -25 "$LOG/$t.log"; fail=1; fi
done
for p in sw2_S0_ctrl_s1.pt sw2_S1_cert03_s1.pt; do
  [ -f "$M/$p" ] || { echo "*** MISSING CHECKPOINT: $p ***"; fail=1; }
done
exit $fail
