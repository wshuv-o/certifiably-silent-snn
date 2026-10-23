#!/usr/bin/env bash
# ACC-012b (pre-registered in research/N3_SCALEUP_PLAN.md). Slot B; waits for ACC-011 where a gate needs it.
source ~/research/venv_gpu/bin/activate
export PYTHONPATH=~/research/refs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=$(cd "$(dirname "$0")" && pwd); LOG=~/research/logs; M=~/research/models
STATUS=$H/../results/acc012b_status.txt
cd ~/research/refs/SCommander-protocol && cp "$H/scmd_rec.py" "$H/scmd_main_rec.py" .
run() { tag=$1; seed=$2; shift 2; f=$LOG/${tag}_s$seed.log
  grep -qas "^RESULT" "$f" && return 0
  echo "--- $tag s$seed $(date +%H:%M:%S) ---"; env TAG=$tag SEED=$seed "$@" python -u scmd_main_rec.py > "$f" 2>&1
  grep -qas "^RESULT" "$f" && echo "OK $tag s$seed $(date +%H:%M:%S)" || { echo "*** FAILED $tag s$seed ***"; tail -15 "$f"; }; }
waitfor() { until [ -f "$LOG/$1.json" ]; do sleep 60; done; }
until [ -f "$LOG/a11_s0.json" ]; do sleep 60; done   # one job at a time: 6 GB card
echo "=== ACC-012b START $(date) ===" | tee "$STATUS"
run a12b_C 0 TEST=0 REC_LR=1e-3
bash "$H/run_acc011.sh" 0   # no-op if seed 0 already finished
G1=$(python - <<'PY'
import json, os
L = os.path.expanduser("~/research/logs")
r = json.load(open(f"{L}/a11_s0.json")); c = json.load(open(f"{L}/a12b_C_s0.json"))
print(f"# a12b_C: val {c['acc_val']*100:.2f}  (SpikCommander a11 s0 val {r['acc_val']*100:.2f}, bar {r['acc_val']*100-1:.2f})  {c['pct_oracle']*100:.1f}% of oracle  R_short {c['R_short']:.3f}")
print("STAGE1 " + ("PASS" if c["acc_val"] >= r["acc_val"] - 0.010 else "FAIL"))
PY
)
echo "$G1" | tee -a "$STATUS"; echo "$G1" | grep -q "STAGE1 PASS" || { echo "=== ACC-012b stage 1 FAIL ===" | tee -a "$STATUS"; exit 0; }
CERT="CERT_LAMBDA=1.0 EPOCHS=100 SELECT_FROM=49 FT_LR=1e-3"
run a12b_Cc 0 $CERT TEST=0 INIT=$M/a12b_C_s0_Best_ACC.pt
G2=$(python - <<'PY'
import json, os
L = os.path.expanduser("~/research/logs")
c = json.load(open(f"{L}/a12b_C_s0.json")); k = json.load(open(f"{L}/a12b_Cc_s0.json")); cost = (k["acc_val"] - c["acc_val"]) * 100
print(f"# a12b_Cc: val {k['acc_val']*100:.2f} ({cost:+.2f})  {k['pct_oracle']*100:.1f}% of oracle  R_short {k['R_short']:.3f}  viol {k['violations']}")
print("STAGE2 " + ("PASS" if k["pct_oracle"] >= 0.90 and k["violations"] == 0 and cost >= -1.5 else "FAIL"))
PY
)
echo "$G2" | tee -a "$STATUS"; echo "$G2" | grep -q "STAGE2 PASS" || { echo "=== ACC-012b stage 2 FAIL ===" | tee -a "$STATUS"; exit 0; }
echo "=== ACC-012b stage 3: TEST $(date) ===" | tee -a "$STATUS"
for s in 1 2 3; do run a12b_T $s TEST=1 REC_LR=1e-3; run a12b_Tc $s $CERT TEST=1 INIT=$M/a12b_T_s${s}_Best_ACC.pt; done
bash "$H/run_acc011.sh" 1; bash "$H/run_acc011.sh" 2   # serial queue: reference seeds for the summary
python - <<'PY' | tee -a "$STATUS"
import json, os
import numpy as np
L = os.path.expanduser("~/research/logs"); J = lambda n: json.load(open(f"{L}/{n}.json"))
ref = [J(f"a11_s{s}") for s in (0, 1, 2)]
rs, rr = np.mean([r["acc_test"] for r in ref]) * 100, np.mean([r["test_best_epoch_rule"] for r in ref]) * 100
print(f"# SpikCommander (ACC-011, seeds 0-2): selected {rs:.2f}  best-test-epoch rule {rr:.2f}")
rows = []
for s in (1, 2, 3):
    c, k = J(f"a12b_T_s{s}"), J(f"a12b_Tc_s{s}"); tc = k["test_cert"]
    rows.append((c["acc_test"]*100, k["acc_test"]*100, tc["pct_oracle"]*100, tc["violations"], c["test_best_epoch_rule"]*100, k["test_best_epoch_rule"]*100))
    r = rows[-1]
    print(f"# seed {s}: control {r[0]:.2f}  certified {r[1]:.2f}  ({r[2]:.1f}% of oracle, viol {r[3]})  [best-test-epoch rule: {r[4]:.2f} / {r[5]:.2f}]")
x = np.array(rows); m = x.mean(0)
print(f"# mean: control {m[0]:.2f}  certified {m[1]:.2f}  {m[2]:.1f}% of oracle  viol {int(x[:,3].sum())}  [best-test-epoch rule: {m[4]:.2f} / {m[5]:.2f}]")
ok = m[1] >= rs - 1.0 and all(r[2] >= 90 for r in rows) and int(x[:, 3].sum()) == 0
print("ACC-012b CONFIRMED" if ok else "ACC-012b NOT CONFIRMED")
PY
echo "=== ACC-012b END $(date) ===" | tee -a "$STATUS"
