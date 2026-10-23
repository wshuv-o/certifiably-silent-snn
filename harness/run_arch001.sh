#!/usr/bin/env bash
# ARCH-001 (pre-registered in research/N3_SCALEUP_PLAN.md before any run): a feedforward front-end for
# the certified recurrent layer. Three stages, gated exactly as pre-registered:
#   1. validation, seed 1: anchor A0, A1 (ffd), A2 (layer). Candidate adopted only at >= +2.0 points.
#   2. validation, seed 1: certificate fine-tune of the candidate. Pass: >= 90% of oracle, 0 violations,
#      cost <= 1.0 point.
#   3. TEST, fresh seeds 2-4, touched once: control + constrained per seed.
# The script stops at the first failed gate and records why. Nothing is selected on the test set.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
STATUS=../results/arch001_status.txt

runone() {
  tag=$1; seed=$2; shift 2
  if grep -qas "^RESULT" "$LOG/ar_${tag}_s${seed}.log"; then echo "SKIP $tag s$seed"; return 0; fi
  echo "--- $tag s$seed $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=ar_$tag SEED=$seed "$@" SAVE=$M/ar_${tag}_s${seed}.pt python -u s5_delays.py > "$LOG/ar_${tag}_s${seed}.log" 2>&1
    if grep -qas "^RESULT" "$LOG/ar_${tag}_s${seed}.log"; then
      sz=$(stat -c%s "$M/ar_${tag}_s${seed}.pt" 2>/dev/null || echo 0)
      [ "$sz" -lt 10000 ] && { echo "*** $tag s$seed: RESULT but checkpoint $sz bytes ***"; return 1; }
      echo "OK $tag s$seed (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag s$seed attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag s$seed ***"; tail -25 "$LOG/ar_${tag}_s${seed}.log"; return 1
}

BASE="H=512 AUG=2 DELAYS=2,4,8 EPOCHS=150"
FED="FE_DELAYS=0,3,6,9,12,15,18,21,24"
A1="FE=ffd $FED"
A2="FE=layer FE_H=256 $FED"
CERT="CERT_LAMBDA=1.0 RAMP=1 LR=5e-4 EPOCHS=150"

echo "=== ARCH-001 START $(date) ===" | tee "$STATUS"

# ---------------- stage 1 ----------------
runone A0 1 $BASE & p0=$!
runone A1 1 $BASE $A1 & p1=$!
runone A2 1 $BASE $A2 & p2=$!
wait $p0 $p1 $p2

DEC=$(python - <<'PY'
import json
r = {a: json.load(open(f"../results/d1_ar_{a}_s1.json")) for a in ("A0", "A1", "A2")}
acc = {a: r[a]["acc_val"] for a in r}
win = max(("A1", "A2"), key=lambda a: acc[a])
gain = acc[win] - acc["A0"]
for a in r:
    x = r[a]; po = x["core_cert"] / x["core_oracle"] if x["core_oracle"] else 0
    print(f"# {a}: val {acc[a]*100:.2f}  cert {x['core_cert']*100:.2f}%  oracle {x['core_oracle']*100:.2f}%  "
          f"({po*100:.1f}% of oracle)  R_short {x['R_short']:.3f}  viol {x['violations']}")
print(f"# candidate {win}: {gain*100:+.2f} points over the same-machine anchor (bar +2.00)")
print("ADOPT " + win if gain >= 0.02 else "NEGATIVE")
PY
)
echo "$DEC" | tee -a "$STATUS"
DECISION=$(echo "$DEC" | tail -1)
if [ "$DECISION" = "NEGATIVE" ]; then
  echo "=== STAGE 1 NEGATIVE: no front-end reaches +2.0 points. ARCH-001 stops here, as pre-registered. ===" | tee -a "$STATUS"
  exit 0
fi
W=$(echo "$DECISION" | awk '{print $2}')
if [ "$W" = "A1" ]; then WCFG="$A1"; else WCFG="$A2"; fi
echo "=== STAGE 1 ADOPTS $W: $WCFG ===" | tee -a "$STATUS"

# ---------------- stage 2 ----------------
runone ${W}cert 1 $BASE $WCFG $CERT INIT=$M/ar_${W}_s1.pt || exit 1
G2=$(W=$W python - <<'PY'
import json, os
W = os.environ["W"]
c = json.load(open(f"../results/d1_ar_{W}_s1.json")); k = json.load(open(f"../results/d1_ar_{W}cert_s1.json"))
po = k["core_cert"] / k["core_oracle"]; cost = (k["acc_val"] - c["acc_val"]) * 100
print(f"# stage 2: constrained val {k['acc_val']*100:.2f} (control {c['acc_val']*100:.2f}, change {cost:+.2f})  "
      f"cert {k['core_cert']*100:.2f}% = {po*100:.1f}% of oracle  R_short {k['R_short']:.3f}  viol {k['violations']}")
print("PASS" if (po >= 0.90 and k["violations"] == 0 and cost >= -1.0) else "FAIL")
PY
)
echo "$G2" | tee -a "$STATUS"
if [ "$(echo "$G2" | tail -1)" != "PASS" ]; then
  echo "=== STAGE 2 FAILED: front-end and certificate incompatible under the established recipe. Stops here. ===" | tee -a "$STATUS"
  exit 0
fi

# ---------------- stage 3 (test, touched once) ----------------
echo "=== STAGE 3: test set, fresh seeds 2-4 $(date) ===" | tee -a "$STATUS"
runone ${W} 2 $BASE $WCFG TEST=1 & q2=$!
runone ${W} 3 $BASE $WCFG TEST=1 & q3=$!
runone ${W} 4 $BASE $WCFG TEST=1 & q4=$!
wait $q2 $q3 $q4
runone ${W}cert 2 $BASE $WCFG $CERT TEST=1 INIT=$M/ar_${W}_s2.pt & r2=$!
runone ${W}cert 3 $BASE $WCFG $CERT TEST=1 INIT=$M/ar_${W}_s3.pt & r3=$!
runone ${W}cert 4 $BASE $WCFG $CERT TEST=1 INIT=$M/ar_${W}_s4.pt & r4=$!
wait $r2 $r3 $r4

W=$W python - <<'PY' | tee -a "$STATUS"
import json, os
import numpy as np
W = os.environ["W"]
rows = []
for s in (2, 3, 4):
    c = json.load(open(f"../results/d1_ar_{W}_s{s}.json")); k = json.load(open(f"../results/d1_ar_{W}cert_s{s}.json"))
    t = k["test"]; po = t["test_core_cert"] / t["test_core_oracle"]
    rows.append((s, c["acc"] * 100, k["acc"] * 100, (k["acc"] - c["acc"]) * 100,
                 c["test"]["test_core_cert"] * 100, t["test_core_cert"] * 100, t["test_core_oracle"] * 100,
                 po * 100, t["test_violations"]))
    print(f"# seed {s}: control {rows[-1][1]:.2f}  constrained {rows[-1][2]:.2f}  cost {rows[-1][3]:+.2f}  "
          f"ctrl cert {rows[-1][4]:.2f}%  cert {rows[-1][5]:.2f}%  oracle {rows[-1][6]:.2f}%  "
          f"({rows[-1][7]:.1f}% of oracle)  viol {rows[-1][8]}")
a = np.array([r[1:] for r in rows], dtype=float)
m = a.mean(0); sd = a[:, 2].std(ddof=1)
print(f"# mean: control {m[0]:.2f}  constrained {m[1]:.2f}  cost {m[2]:+.2f} (sd {sd:.2f})  "
      f"cert {m[4]:.2f}% ({m[6]:.1f}% of oracle)  viol {int(a[:, 7].sum())}")
ok = m[1] >= 87.59 + 1.5 and all(r[7] >= 90 for r in rows) and int(a[:, 7].sum()) == 0
print("CONFIRMED" if ok else "NOT CONFIRMED")
PY
echo "=== ARCH-001 END $(date) ===" | tee -a "$STATUS"
