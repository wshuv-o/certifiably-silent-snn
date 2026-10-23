#!/usr/bin/env bash
# ACC-007 (pre-registered in research/N3_SCALEUP_PLAN.md). Runs inside the reference code copy.
source ~/research/venv_gpu/bin/activate
export PYTHONPATH=~/research/refs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=$(cd "$(dirname "$0")" && pwd); LOG=~/research/logs; M=~/research/models
STATUS=$H/../results/acc007_status.txt
cd ~/research/refs/SNN-delays-protocol && cp "$H/acc005_rec.py" "$H/acc005_main.py" .

run() {  # tag seed env...
  tag=$1; seed=$2; shift 2; f=$LOG/${tag}_s$seed.log
  grep -qas "^RESULT" "$f" && { echo "SKIP $tag s$seed"; return 0; }
  echo "--- $tag s$seed $(date +%H:%M:%S) ---"
  env TAG=$tag SEED=$seed "$@" python -u acc005_main.py > "$f" 2>&1
  grep -qas "^RESULT" "$f" && echo "OK $tag s$seed $(date +%H:%M:%S)" || { echo "*** FAILED $tag s$seed ***"; tail -15 "$f"; }
}
J() { python -c "import json,sys; print(json.load(open('$LOG/'+sys.argv[1]+'.json'))$2)" "$1"; }

echo "=== ACC-007 START $(date) ===" | tee "$STATUS"
W=C1; D=0.4
CERT="CERT_LAMBDA=1.0 RAMP=1 EPOCHS=100 SELECT=best_val SELECT_FROM=49"
run a7_C1c 0 REC_DROP=$D $CERT INIT=$M/a5_C1_s0_Best_ACC.pt
G=$(python - <<'PY'
import json, os
L = os.path.expanduser("~/research/logs")
c = json.load(open(f"{L}/a5_C1_s0.json")); k = json.load(open(f"{L}/a7_C1c_s0.json"))
cost = (k["acc_val"] - c["acc_val"]) * 100
print(f"# a7_C1c: val {k['acc_val']*100:.2f} (control {c['acc_val']*100:.2f}, {cost:+.2f})  {k['pct_oracle']*100:.1f}% of oracle  "
      f"R_short {k['R_short']:.3f}  viol {k['violations']}  selected {k['selected']}")
print("STAGE2 " + ("PASS" if k["pct_oracle"] >= 0.90 and k["violations"] == 0 and cost >= -1.5 else "FAIL"))
PY
)
echo "$G" | tee -a "$STATUS"
echo "$G" | grep -q "^STAGE2 PASS" || { echo "=== ACC-007 stage 2 FAIL ===" | tee -a "$STATUS"; exit 0; }
# stage 3: TEST, fresh seeds 1-3 (two at a time: 6 GB card)
echo "=== ACC-007 stage 3: TEST $(date) ===" | tee -a "$STATUS"
# 6 GB card: two unpenalised jobs fit together; a penalised job runs alone
run a7_${W}T 1 REC_DROP=$D TEST=1 & a=$!; run a7_${W}T 2 REC_DROP=$D TEST=1 & b=$!; wait $a $b
run a7_${W}T 3 REC_DROP=$D TEST=1
for s in 1 2 3; do run a7_${W}Tc $s REC_DROP=$D $CERT TEST=1 INIT=$M/a7_${W}T_s${s}_Best_ACC.pt; done
W=$W python - <<'PY' | tee -a "$STATUS"
import json, os, re
import numpy as np
L = os.path.expanduser("~/research/logs"); W = os.environ["W"]; rows = []
def best_test(tag, s):
    t = [float(m) for m in re.findall(r"Acc Test = ([0-9.]+)%", open(f"{L}/{tag}_s{s}.log").read())]
    return max(t) if t else float("nan")
for s in (1, 2, 3):
    c = json.load(open(f"{L}/a7_{W}T_s{s}.json")); k = json.load(open(f"{L}/a7_{W}Tc_s{s}.json"))
    tc = k["test_cert"]
    rows.append((c["acc_test"]*100, k["acc_test"]*100, tc["pct_oracle"]*100, tc["violations"],
                 best_test(f"a7_{W}T", s), best_test(f"a7_{W}Tc", s)))
    r = rows[-1]
    print(f"# seed {s}: control {r[0]:.2f}  certified {r[1]:.2f}  ({r[2]:.1f}% of oracle, viol {r[3]})  "
          f"[best-test-epoch rule: control {r[4]:.2f}, certified {r[5]:.2f}]")
x = np.array(rows); m = np.nanmean(x, 0)
print(f"# mean: control {m[0]:.2f}  certified {m[1]:.2f}  {m[2]:.1f}% of oracle  viol {int(x[:,3].sum())}  "
      f"[best-test-epoch rule: control {m[4]:.2f}, certified {m[5]:.2f}]")
ok = m[1] >= 90.0 and all(r[2] >= 90 for r in rows) and int(x[:, 3].sum()) == 0
print("ACC-007 CONFIRMED" if ok else "ACC-007 NOT CONFIRMED")
PY
echo "=== ACC-007 END $(date) ===" | tee -a "$STATUS"
