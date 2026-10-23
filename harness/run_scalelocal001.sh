#!/usr/bin/env bash
# SCALE-LOCAL-001 (pre-registered): is certifiability governed by fan-in rather than by width?
#
# R_i sums the positive recurrent weights into neuron i, so it scales with fan-in. Under dense
# recurrence fan-in = H and R grows with width (control R_mean 4.78 at H=512 -> 8.72 at H=1024),
# which is why SCALE-FT-001 failed at H=1024. The ring mask here keeps fan-in at
# (2*LOCAL_R+1)*CPC = 96 neurons at every width, so if fan-in is what matters, certification
# should be width-independent.
#
#   usage: run_scalelocal001.sh <A|B|C>
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
CH=$1

runone() {
  tag=$1; shift
  if grep -qas "^RESULT" "$LOG/sl_$tag.log"; then echo "[$CH] SKIP $tag"; return 0; fi
  echo "[$CH] --- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/sl_${tag}.pt python -u s5_delays.py > "$LOG/sl_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/sl_$tag.log"; then
      sz=$(stat -c%s "$M/sl_${tag}.pt" 2>/dev/null || echo 0)
      if [ "$sz" -lt 10000 ]; then echo "[$CH] *** $tag: RESULT but checkpoint $sz bytes ***"; return 1; fi
      echo "[$CH] OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "[$CH] retry $tag attempt $a"; sleep 10
  done
  echo "[$CH] *** FAILED: $tag ***"; tail -20 "$LOG/sl_$tag.log"; return 1
}

# LOCAL_R=1 with CPC=32 holds fan-in at 96 neurons for every H
LOC="DELAYS=2,4,8 AUG=2 EPOCHS=150 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1"
DEN="DELAYS=2,4,8 AUG=2 EPOCHS=150 SEED=1 CPC=32"
FT="LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 EPOCHS=150"

echo "[$CH] START $(date +%H:%M:%S)"
case "$CH" in
  A)  # the two cheaper widths, control then its matched constrained arm
    runone lc512   $LOC H=512
    runone lk512   DELAYS=2,4,8 AUG=2 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1 H=512  INIT=$M/sl_lc512.pt  $FT
    runone lc1024  $LOC H=1024
    runone lk1024  DELAYS=2,4,8 AUG=2 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1 H=1024 INIT=$M/sl_lc1024.pt $FT
    ;;
  B)  # the decisive width; compute scales as H^2 so these are the long runs
    runone lc2048  $LOC H=2048
    runone lk2048  DELAYS=2,4,8 AUG=2 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1 H=2048 INIT=$M/sl_lc2048.pt $FT
    ;;
  C)  # dense contrast at the top width: fan-in = H = 2048 here
    runone dc2048  $DEN H=2048
    ;;
esac
echo "[$CH] DONE $(date +%H:%M:%S)"
