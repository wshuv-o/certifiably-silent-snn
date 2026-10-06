#!/usr/bin/env bash
# CONFIRM-002: SWEEP-002 winner on seeds 2-4 (parallel), test set. Ours first unless ours needs that seed's control.
# Logging note (2026-10-06): full logs to ~/research/logs, no grep pipe (it discarded tracebacks and
# hid a failed torch.save). Each chain aborts if a checkpoint a later run depends on is missing.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
WTAG=$(python -c "import json;print(json.load(open('../results/sw2_winner.json'))['tag'])")
WENV=$(python -c "import json;print(json.load(open('../results/sw2_winner.json'))['env'])")
echo "winner: $WTAG  env: $WENV"
run() {
  tag=$1; shift
  env TEST=1 TAG=$tag "$@" SAVE=$M/c2_${tag}.pt python -u s4_improve.py > "$LOG/c2_$tag.log" 2>&1
  if grep -q "^RESULT" "$LOG/c2_$tag.log"; then grep -h "^RESULT" "$LOG/c2_$tag.log"
  else echo "*** FAILED: $tag ***"; tail -25 "$LOG/c2_$tag.log"; return 1; fi
  [ -f "$M/c2_${tag}.pt" ] || { echo "*** MISSING CHECKPOINT: c2_${tag}.pt ***"; return 1; }
}
chain() {
  S=$1; CTRL=$M/c2_ctrl_s$S.pt; E=${WENV//\{CTRL\}/$CTRL}
  if [[ "$WENV" == *"{CTRL}"* ]]; then
    run ctrl_s$S SEED=$S || { echo "*** chain seed $S aborted: control failed, 'ours' depends on it ***"; return 1; }
    run ours_s$S SEED=$S $E || echo "*** seed $S: ours failed ***"
  else
    run ours_s$S SEED=$S $E || echo "*** seed $S: ours failed ***"
    run ctrl_s$S SEED=$S     || echo "*** seed $S: control failed ***"
  fi
  run ref_cert03_s$S SEED=$S CERT_LAMBDA=0.3 || echo "*** seed $S: reference failed ***"
}
for S in 5 6 7; do chain $S & done
wait
