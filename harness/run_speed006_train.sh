#!/usr/bin/env bash
# SPEED-006 training (pre-registered): K = 8 certificate fine-tune of the local control.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"; M=~/research/models; LOG=~/research/logs
until [ -f ../results/d1_sp5_lc512_s1.json ] && [ -s $M/sp5_lc512.pt ]; do sleep 60; done
f=$LOG/sp6_lk8.log
if ! grep -qas "^RESULT" "$f"; then
  echo "--- lk8 $(date +%H:%M:%S) ---"
  TAG=sp6_lk8 DELAYS=2,4,8 AUG=2 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1 H=512 INIT=$M/sp5_lc512.pt LR=5e-4 RAMP=1 \
    CERT_LAMBDA=1.0 K=8 EPOCHS=150 SAVE=$M/sp6_lk8.pt python -u s5_delays.py > "$f" 2>&1
fi
grep -qas "^RESULT" "$f" && echo "OK lk8 $(date +%H:%M:%S)" || { echo "*** FAILED lk8 ***"; tail -15 "$f"; }
python - <<'PY'
import json
c = json.load(open("../results/d1_sp5_lc512_s1.json")); k = json.load(open("../results/d1_sp6_lk8_s1.json"))
print(f"# lk8 (K=8): val {k['acc_val']*100:.2f} (control {c['acc_val']*100:.2f}, {(k['acc_val']-c['acc_val'])*100:+.2f})  "
      f"{k['core_cert']/k['core_oracle']*100:.1f}% of oracle at K=8  R_short {k['R_short']:.3f}  viol {k['violations']}")
PY
echo "=== SPEED-006 TRAINING END $(date) ==="
