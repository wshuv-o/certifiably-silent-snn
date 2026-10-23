#!/usr/bin/env bash
# GOVERN-003 (pre-registered in research/N3_SCALEUP_PLAN.md before any run): penalty-weight sweep from the
# same-machine anchor, to sample the transition of the R_short relation. Validation only.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
runone() {
  tag=$1; shift
  if grep -qas "^RESULT" "$LOG/g3_$tag.log"; then echo "SKIP $tag"; return 0; fi
  for a in 1 2 3; do
    env TAG=g3_$tag SEED=1 "$@" SAVE=$M/g3_${tag}.pt python -u s5_delays.py > "$LOG/g3_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/g3_$tag.log"; then
      sz=$(stat -c%s "$M/g3_${tag}.pt" 2>/dev/null || echo 0)
      [ "$sz" -lt 10000 ] && { echo "*** $tag: checkpoint $sz bytes ***"; return 1; }
      echo "OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag ***"; tail -20 "$LOG/g3_$tag.log"; return 1
}
BASE="H=512 AUG=2 DELAYS=2,4,8 RAMP=1 LR=5e-4 EPOCHS=150 INIT=$M/ar_A0_s1.pt"
echo "=== GOVERN-003 START $(date) ==="
pids=""
for lam in 0.001 0.003 0.01 0.03 0.1; do
  runone lam$lam $BASE CERT_LAMBDA=$lam & pids="$pids $!"
done
wait $pids
python - <<'PY'
import json
import numpy as np
a = json.load(open("../results/d1_ar_A0_s1.json"))
rows = [("anchor", 0.0, a)] + [(f"lam{l}", l, json.load(open(f"../results/d1_g3_lam{l}_s1.json")))
                               for l in (0.001, 0.003, 0.01, 0.03, 0.1)]
for tag, lam, r in rows:
    print(f"# {tag:9s} lambda {lam:<6g} val {r['acc_val']*100:6.2f}  R_short {r['R_short']:6.3f}  "
          f"cert {r['core_cert']*100:6.2f}%  oracle {r['core_oracle']*100:6.2f}%  "
          f"{r['core_cert']/r['core_oracle']*100:5.1f}% of oracle  viol {r['violations']}")
rs = np.array([r["R_short"] for _, _, r in rows]); po = np.array([r["core_cert"] / r["core_oracle"] for _, _, r in rows])
rk = lambda v: np.argsort(np.argsort(v))
print(f"# Spearman(R_short, % of oracle) over {len(rows)} networks: {np.corrcoef(rk(rs), rk(po))[0,1]:.3f}")
print(f"# arms in 0.3-2.4: {int(((rs > 0.3) & (rs < 2.4)).sum())}")
PY
echo "=== GOVERN-003 END $(date) ==="
