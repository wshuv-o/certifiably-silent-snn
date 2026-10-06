#!/usr/bin/env bash
# CONFIRM-002: SWEEP-002 winner on seeds 2-4 (parallel), test set. Ours first unless ours needs that seed's control.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models
WTAG=$(python -c "import json;print(json.load(open('../results/sw2_winner.json'))['tag'])")
WENV=$(python -c "import json;print(json.load(open('../results/sw2_winner.json'))['env'])")
echo "winner: $WTAG  env: $WENV"
run() { tag=$1; shift; env TEST=1 TAG=$tag "$@" SAVE=$M/c2_${tag}.pt python -u s4_improve.py 2>&1 | grep --line-buffered -E "^(RESULT)"; }
chain() {
  S=$1; CTRL=$M/c2_ctrl_s$S.pt; E=${WENV//\{CTRL\}/$CTRL}
  if [[ "$WENV" == *"{CTRL}"* ]]; then
    run ctrl_s$S SEED=$S; run ours_s$S SEED=$S $E          # dependency: ours starts from this seed's control
  else
    run ours_s$S SEED=$S $E; run ctrl_s$S SEED=$S          # ours first
  fi
  run ref_cert03_s$S SEED=$S CERT_LAMBDA=0.3               # original method (reference), last
}
for S in 2 3 4; do chain $S & done
wait
