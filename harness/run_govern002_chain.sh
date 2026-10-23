#!/usr/bin/env bash
# GOVERN-002, run as three independent chains so the remaining work overlaps.
#
# Measured on this machine during a single run: GPU 23-28% utilised, 1.9 of 16 GB, 84 W of a ~360 W
# budget, with the Python process at 410% CPU. The workload is launch-latency bound, so three
# concurrent chains have headroom on both the GPU and the 32 CPU threads.
#
# Three is the documented safe ceiling for concurrent CUDA processes here; six previously caused
# CUDA_ERROR_UNKNOWN and destroyed runs. Each run retries three times and any run whose log already
# holds a RESULT line is skipped, so a driver fault costs one run rather than the batch.
#
#   usage: run_govern002_chain.sh <1|2|3>
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
CHAIN=$1

runone() {
  tag=$1; script=$2; shift 2
  if grep -qas "^RESULT" "$LOG/g2_$tag.log"; then echo "[c$CHAIN] SKIP $tag"; return 0; fi
  echo "[c$CHAIN] --- $tag $(date +%H:%M:%S) ---"
  for a in 1 2 3; do
    env TAG=$tag "$@" SAVE=$M/g2_${tag}.pt python -u "$script" > "$LOG/g2_$tag.log" 2>&1
    if grep -qas "^RESULT" "$LOG/g2_$tag.log"; then
      sz=$(stat -c%s "$M/g2_${tag}.pt" 2>/dev/null || echo 0)
      if [ "$sz" -lt 10000 ]; then echo "[c$CHAIN] *** $tag: RESULT but checkpoint $sz bytes ***"; return 1; fi
      echo "[c$CHAIN] OK $tag (attempt $a) $(date +%H:%M:%S)"; return 0
    fi
    echo "[c$CHAIN] retry $tag attempt $a"; sleep 10
  done
  echo "[c$CHAIN] *** FAILED: $tag ***"; tail -20 "$LOG/g2_$tag.log"; return 1
}

COMMON="H=512 AUG=2 EPOCHS=150"
FT="LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 EPOCHS=150"

echo "[c$CHAIN] START $(date +%H:%M:%S)"
case "$CHAIN" in
  1)  # no fine-tune dependencies in this chain
    runone u1_s3  s5_delays.py $COMMON SEED=3 DELAYS=1
    runone m4_s2  s5_delays.py $COMMON SEED=2 DELAYS=1,2,4,8
    runone m4_s3  s5_delays.py $COMMON SEED=3 DELAYS=1,2,4,8
    ;;
  2)  # each constrained arm fine-tunes from the control of the same seed, so order matters
    runone t3c_s2 s5_delays.py $COMMON SEED=2 DELAYS=2,4,8
    runone t3k_s2 s5_delays.py H=512 AUG=2 SEED=2 DELAYS=2,4,8 INIT=$M/g2_t3c_s2.pt $FT
    runone t3c_s3 s5_delays.py $COMMON SEED=3 DELAYS=2,4,8
    runone t3k_s3 s5_delays.py H=512 AUG=2 SEED=3 DELAYS=2,4,8 INIT=$M/g2_t3c_s3.pt $FT
    ;;
  3)
    runone dcc_s2 s7_dcls.py $COMMON SEED=2 DMIN=2 DMAX=8
    runone dck_s2 s7_dcls.py H=512 AUG=2 SEED=2 DMIN=2 DMAX=8 INIT=$M/g2_dcc_s2.pt $FT
    runone dcc_s3 s7_dcls.py $COMMON SEED=3 DMIN=2 DMAX=8
    runone dck_s3 s7_dcls.py H=512 AUG=2 SEED=3 DMIN=2 DMAX=8 INIT=$M/g2_dcc_s3.pt $FT
    ;;
esac
echo "[c$CHAIN] DONE $(date +%H:%M:%S)"
