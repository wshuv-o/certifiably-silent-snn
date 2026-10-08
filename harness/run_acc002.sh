#!/usr/bin/env bash
# ACC-002 (pre-registered in N3_SCALEUP_PLAN.md): per-neuron learnable membrane time constants.
#
# The gate comes first. beta enters the certificate in five places and was a scalar in all of them;
# the patch makes it a shape-(H,) tensor that broadcasts. If TAU_LEARN=0 does not reproduce the known
# five-epoch value to every digit, the broadcast changed the scalar path and nothing after it can be
# trusted, so the script stops.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"

EXPECT=0.602224123182207

echo "=== ACC-002 gate: TAU_LEARN=0 must reproduce $EXPECT ==="
env TAG=t_eqcheck H=512 SEED=1 AUG=2 DELAYS=2,4,8 EPOCHS=5 python -u s5_delays.py > "$LOG/ac_t_eqcheck.log" 2>&1
got=$(grep -a "^RESULT" "$LOG/ac_t_eqcheck.log" | python3 -c "import json,sys; print(repr(json.loads(sys.stdin.read()[7:])['acc_val']))" 2>/dev/null)
echo "expected $EXPECT"
echo "got      $got"
if [ "$got" != "$EXPECT" ]; then
  echo "*** GATE FAILED: the scalar path changed. ACC-002 aborted. ***"
  tail -20 "$LOG/ac_t_eqcheck.log"
  exit 1
fi
echo "gate passed"

runone() {
  tag=$1; shift
  if grep -qas "^RESULT" "$LOG/ac_$tag.log"; then echo "SKIP $tag"; return 0; fi
  echo "--- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=ac_$tag "$@" SAVE=$M/ac_${tag}.pt python -u s5_delays.py > "$LOG/ac_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/ac_$tag.log"; then
      sz=$(stat -c%s "$M/ac_${tag}.pt" 2>/dev/null || echo 0)
      [ "$sz" -lt 10000 ] && { echo "*** $tag: RESULT but checkpoint $sz bytes ***"; return 1; }
      echo "OK $tag $(date +%H:%M:%S)"; return 0
    fi
    echo "retry $tag attempt $a"; sleep 10
  done
  echo "*** FAILED: $tag ***"; tail -20 "$LOG/ac_$tag.log"; return 1
}

BASE="H=512 SEED=1 AUG=2 DELAYS=2,4,8 EPOCHS=150"

# control: does learning tau raise accuracy at all, and do the time constants actually move?
runone t_ctrl $BASE TAU_LEARN=1 || exit 1
# and can the certificate still be recovered on top of it, at the usual cost?
runone t_cert $BASE TAU_LEARN=1 INIT=$M/ac_t_ctrl.pt LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 || exit 1

echo "=== ACC-002 COMPLETE $(date) ==="
