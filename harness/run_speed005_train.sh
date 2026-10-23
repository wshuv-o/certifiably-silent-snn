#!/usr/bin/env bash
# SPEED-005 training (pre-registered): locally connected control and certified fine-tune, recipe of tab:scale.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"; M=~/research/models; LOG=~/research/logs
until [ -f "$LOG/a12c_Tc_s3.json" ]; do sleep 60; done           # after the last ACC-012c test seed
L="DELAYS=2,4,8 AUG=2 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1 H=512"
run() { tag=$1; shift; f=$LOG/sp5_$tag.log; grep -qas "^RESULT" "$f" && return 0
  echo "--- $tag $(date +%H:%M:%S) ---"; env TAG=sp5_$tag "$@" SAVE=$M/sp5_$tag.pt python -u s5_delays.py > "$f" 2>&1
  grep -qas "^RESULT" "$f" && echo "OK $tag $(date +%H:%M:%S)" || { echo "*** FAILED $tag ***"; tail -15 "$f"; }; }
run lc512 $L EPOCHS=150
run lk512 $L INIT=$M/sp5_lc512.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 EPOCHS=150
python - <<'PY'
import json
for t in ("lc512", "lk512"):
    x = json.load(open(f"../results/d1_sp5_{t}_s1.json"))
    print(f"# {t}: val {x['acc_val']*100:.2f}  {x['core_cert']/x['core_oracle']*100:.1f}% of oracle  R_short {x['R_short']:.3f}  viol {x['violations']}")
PY
echo "=== SPEED-005 TRAINING END $(date) ==="
