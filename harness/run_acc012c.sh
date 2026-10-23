#!/usr/bin/env bash
# ACC-012c stage 3 (pre-registered): ACC-012b recipe, adaptive certificate primary, plain recorded. Serial.
source ~/research/venv_gpu/bin/activate
export PYTHONPATH=~/research/refs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CERT_ADAPT=1
H=$(cd "$(dirname "$0")" && pwd); LOG=~/research/logs; M=~/research/models
STATUS=$H/../results/acc012c_status.txt
cd ~/research/refs/SCommander-protocol && cp "$H/scmd_rec.py" "$H/scmd_main_rec.py" .
run() { tag=$1; seed=$2; shift 2; f=$LOG/${tag}_s$seed.log
  grep -qas "^RESULT" "$f" && return 0
  echo "--- $tag s$seed $(date +%H:%M:%S) ---"; env TAG=$tag SEED=$seed "$@" python -u scmd_main_rec.py > "$f" 2>&1
  grep -qas "^RESULT" "$f" && echo "OK $tag s$seed $(date +%H:%M:%S)" || { echo "*** FAILED $tag s$seed ***"; tail -15 "$f"; }; }
echo "=== ACC-012c START $(date) ===" | tee "$STATUS"
python - <<'PY' | tee -a "$STATUS"
import json, os
L = os.path.expanduser("~/research/logs")
c = json.load(open(f"{L}/a12b_C_s0.json")); k = json.load(open(f"{L}/a12b_Cc_s0.json")); r = json.load(open(f"{L}/a12c_recert_Cc_s0.json"))
cost = (k["acc_val"] - c["acc_val"]) * 100; a = r["adaptive"]
print(f"# stage 2 (re-certified): val {k['acc_val']*100:.2f} ({cost:+.2f})  adaptive {a['pct_oracle']*100:.1f}% of oracle viol {a['violations']}  (plain {r['plain']['pct_oracle']*100:.1f}%)")
print("STAGE2 " + ("PASS" if a["pct_oracle"] >= 0.90 and a["violations"] == 0 and cost >= -1.5 else "FAIL"))
PY
grep -q "STAGE2 PASS" "$STATUS" || { echo "=== ACC-012c stage 2 FAIL ===" | tee -a "$STATUS"; exit 0; }
echo "=== ACC-012c stage 3: TEST $(date) ===" | tee -a "$STATUS"
CERT="CERT_LAMBDA=1.0 EPOCHS=100 SELECT_FROM=49 FT_LR=1e-3"
for s in 1 2 3; do run a12c_T $s TEST=1 REC_LR=1e-3; run a12c_Tc $s $CERT TEST=1 INIT=$M/a12c_T_s${s}_Best_ACC.pt; done
bash "$H/run_acc011.sh" 1; bash "$H/run_acc011.sh" 2
python - <<'PY' | tee -a "$STATUS"
import json, os
import numpy as np
L = os.path.expanduser("~/research/logs"); J = lambda n: json.load(open(f"{L}/{n}.json"))
ref = [J(f"a11_s{s}") for s in (0, 1, 2)]
rs, rr = np.mean([r["acc_test"] for r in ref]) * 100, np.mean([r["test_best_epoch_rule"] for r in ref]) * 100
print(f"# SpikCommander (ACC-011, seeds 0-2): selected {rs:.2f}  best-test-epoch rule {rr:.2f}  per seed {[round(r['acc_test']*100,2) for r in ref]}")
rows = []
for s in (1, 2, 3):
    c, k = J(f"a12c_T_s{s}"), J(f"a12c_Tc_s{s}"); tc, tp = k["test_cert"], k["test_cert_other"]
    rows.append((c["acc_test"]*100, k["acc_test"]*100, tc["pct_oracle"]*100, tc["violations"], tp["pct_oracle"]*100, tp["violations"],
                 c["test_best_epoch_rule"]*100, k["test_best_epoch_rule"]*100))
    r = rows[-1]
    print(f"# seed {s}: control {r[0]:.2f}  certified {r[1]:.2f}  adaptive {r[2]:.1f}% of oracle viol {r[3]}  (plain {r[4]:.1f}% viol {r[5]})  [best-test-epoch rule {r[6]:.2f} / {r[7]:.2f}]")
x = np.array(rows); m = x.mean(0)
print(f"# mean: control {m[0]:.2f}  certified {m[1]:.2f}  adaptive {m[2]:.1f}% (plain {m[4]:.1f}%)  viol {int(x[:,3].sum())}/{int(x[:,5].sum())}  [best-test-epoch rule {m[6]:.2f} / {m[7]:.2f}]")
ok = m[1] >= rs - 1.0 and all(r[2] >= 90 for r in rows) and int(x[:, 3].sum()) == 0 and int(x[:, 5].sum()) == 0
print("ACC-012c CONFIRMED" if ok else "ACC-012c NOT CONFIRMED")
PY
echo "=== ACC-012c END $(date) ===" | tee -a "$STATUS"
