#!/usr/bin/env bash
# ACC-003 + SPEED-003 training (pre-registered in research/N3_SCALEUP_PLAN.md before any run).
#   stage 1 (validation, seed 1): B1, B2 (DCLS front-end, dropout 0.4 / 0), W0 (D={2,4,8} -> {2,8,16}, K=8)
#   stage 2 (validation, seed 1): certificate fine-tunes -- ACC winner (if adopted), W1 (K=8), R1 (A0 at lambda 1.0)
#   stage 3 (TEST, seeds 2-4):    ACC winner control + constrained, only if stages 1-2 pass
# The engine measurement for SPEED-003 needs an idle CPU and runs separately (run_speed003.sh).
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
STATUS=../results/acc003_status.txt

runone() {
  tag=$1; seed=$2; shift 2
  if grep -qas "^RESULT" "$LOG/a3_${tag}_s${seed}.log"; then echo "SKIP $tag s$seed"; return 0; fi
  echo "--- $tag s$seed $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=a3_$tag SEED=$seed "$@" SAVE=$M/a3_${tag}_s${seed}.pt python -u s5_delays.py > "$LOG/a3_${tag}_s${seed}.log" 2>&1
    if grep -qas "^RESULT" "$LOG/a3_${tag}_s${seed}.log"; then
      sz=$(stat -c%s "$M/a3_${tag}_s${seed}.pt" 2>/dev/null || echo 0)
      [ "$sz" -lt 10000 ] && { echo "*** $tag s$seed: checkpoint $sz bytes ***"; return 1; }
      echo "OK $tag s$seed (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag s$seed attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag s$seed ***"; tail -25 "$LOG/a3_${tag}_s${seed}.log"; return 1
}

BASE="H=512 AUG=2 EPOCHS=150"
D3="DELAYS=2,4,8"
DW="DELAYS=2,8,16 K=8"
FEC="FE=dcls NBINS=5 FE_LAYERS=2 FE_H=256 FE_DMAX=25"
CERT="CERT_LAMBDA=1.0 RAMP=1 LR=5e-4"

echo "=== ACC-003 / SPEED-003 START $(date) ===" | tee "$STATUS"

# ---------------- stage 1 ----------------
runone B1 1 $BASE $D3 $FEC FE_DROP=0.4 & p1=$!
runone B2 1 $BASE $D3 $FEC FE_DROP=0   & p2=$!
runone W0 1 $BASE $DW                  & p3=$!
wait $p1 $p2 $p3

DEC=$(python - <<'PY'
import json
a0 = json.load(open("../results/d1_ar_A0_s1.json"))["acc_val"]
r = {k: json.load(open(f"../results/d1_a3_{k}_s1.json")) for k in ("B1", "B2", "W0")}
for k, x in r.items():
    print(f"# {k}: val {x['acc_val']*100:.2f}  cert {x['core_cert']*100:.2f}%  oracle {x['core_oracle']*100:.2f}%  "
          f"({x['core_cert']/x['core_oracle']*100:.1f}% of oracle)  R_short {x['R_short']:.3f}  viol {x['violations']}")
w = max(("B1", "B2"), key=lambda k: r[k]["acc_val"]); g = (r[w]["acc_val"] - a0) * 100
print(f"# ACC candidate {w}: {g:+.2f} points over the anchor 86.40 (bar +3.00)")
print("ADOPT " + w if g >= 3.0 else "NEGATIVE")
PY
)
echo "$DEC" | tee -a "$STATUS"
ACCW=""
if [ "$(echo "$DEC" | tail -1 | cut -d' ' -f1)" = "ADOPT" ]; then
  ACCW=$(echo "$DEC" | tail -1 | cut -d' ' -f2)
  [ "$ACCW" = "B1" ] && ACCCFG="$FEC FE_DROP=0.4" || ACCCFG="$FEC FE_DROP=0"
  echo "=== ACC stage 1 ADOPTS $ACCW ===" | tee -a "$STATUS"
else
  echo "=== ACC stage 1 NEGATIVE: no front-end reaches +3.0. ACC-003 stops, as pre-registered. ===" | tee -a "$STATUS"
fi

# ---------------- stage 2 ----------------
runone W1 1 $BASE $DW $CERT INIT=$M/a3_W0_s1.pt & q1=$!
runone R1 1 $BASE $D3 $CERT INIT=$M/ar_A0_s1.pt & q2=$!
if [ -n "$ACCW" ]; then runone ${ACCW}c 1 $BASE $D3 $ACCCFG $CERT INIT=$M/a3_${ACCW}_s1.pt & q3=$!; wait $q3; fi
wait $q1 $q2

G2=$(ACCW="$ACCW" python - <<'PY'
import json, os
def L(t): return json.load(open(f"../results/d1_a3_{t}_s1.json"))
w0, w1, r1 = L("W0"), L("W1"), L("R1")
po = lambda x: x["core_cert"] / x["core_oracle"]
print(f"# W1 (D={{2,8,16}}, K=8): val {w1['acc_val']*100:.2f} (W0 {w0['acc_val']*100:.2f})  "
      f"{po(w1)*100:.1f}% of oracle  R_short {w1['R_short']:.3f}  viol {w1['violations']}")
print(f"# R1 (D={{2,4,8}}, K=4): val {r1['acc_val']*100:.2f}  {po(r1)*100:.1f}% of oracle  R_short {r1['R_short']:.3f}  viol {r1['violations']}")
print("SPEED_TRAIN " + ("PASS" if (po(w1) >= 0.90 and w1["violations"] == 0 and w1["acc_val"] >= 0.8490) else "FAIL"))
a = os.environ.get("ACCW")
if a:
    c, k = L(a), L(a + "c"); cost = (k["acc_val"] - c["acc_val"]) * 100
    print(f"# ACC {a}c: val {k['acc_val']*100:.2f} (control {c['acc_val']*100:.2f}, change {cost:+.2f})  "
          f"{po(k)*100:.1f}% of oracle  R_short {k['R_short']:.3f}  viol {k['violations']}")
    print("ACC_STAGE2 " + ("PASS" if (po(k) >= 0.90 and k["violations"] == 0 and cost >= -1.0) else "FAIL"))
PY
)
echo "$G2" | tee -a "$STATUS"

# ---------------- stage 3 (ACC only, test touched once) ----------------
if [ -n "$ACCW" ] && echo "$G2" | grep -q "^ACC_STAGE2 PASS"; then
  echo "=== ACC stage 3: TEST, seeds 2-4 $(date) ===" | tee -a "$STATUS"
  runone $ACCW 2 $BASE $D3 $ACCCFG TEST=1 & s2=$!
  runone $ACCW 3 $BASE $D3 $ACCCFG TEST=1 & s3=$!
  runone $ACCW 4 $BASE $D3 $ACCCFG TEST=1 & s4=$!
  wait $s2 $s3 $s4
  runone ${ACCW}c 2 $BASE $D3 $ACCCFG $CERT TEST=1 INIT=$M/a3_${ACCW}_s2.pt & t2=$!
  runone ${ACCW}c 3 $BASE $D3 $ACCCFG $CERT TEST=1 INIT=$M/a3_${ACCW}_s3.pt & t3=$!
  runone ${ACCW}c 4 $BASE $D3 $ACCCFG $CERT TEST=1 INIT=$M/a3_${ACCW}_s4.pt & t4=$!
  wait $t2 $t3 $t4
  ACCW=$ACCW python - <<'PY' | tee -a "$STATUS"
import json, os
import numpy as np
a = os.environ["ACCW"]; rows = []
for s in (2, 3, 4):
    c = json.load(open(f"../results/d1_a3_{a}_s{s}.json")); k = json.load(open(f"../results/d1_a3_{a}c_s{s}.json"))
    t = k["test"]; po = t["test_core_cert"] / t["test_core_oracle"] * 100
    rows.append((c["acc"] * 100, k["acc"] * 100, po, t["test_violations"]))
    print(f"# seed {s}: control {rows[-1][0]:.2f}  constrained {rows[-1][1]:.2f}  cost {rows[-1][1]-rows[-1][0]:+.2f}  "
          f"{po:.1f}% of oracle  viol {rows[-1][3]}")
x = np.array(rows); m = x.mean(0)
print(f"# mean: control {m[0]:.2f}  constrained {m[1]:.2f}  cost {m[1]-m[0]:+.2f}  {m[2]:.1f}% of oracle  viol {int(x[:,3].sum())}")
ok = m[1] >= 90.59 and all(r[2] >= 90 for r in rows) and int(x[:, 3].sum()) == 0
print("ACC CONFIRMED" if ok else "ACC NOT CONFIRMED")
PY
fi
echo "=== ACC-003 / SPEED-003 TRAINING END $(date) ===" | tee -a "$STATUS"
