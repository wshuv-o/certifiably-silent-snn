#!/usr/bin/env bash
# Keep the GPU at three concurrent training processes and no more.
#
# Three is a hard limit on this machine: six caused CUDA_ERROR_UNKNOWN and destroyed five runs, and
# measurement showed no throughput gain above three because the workload is kernel-launch bound.
#
# The pgrep pattern uses a bracketed character class so the pattern cannot match this script's own
# command line -- a plain pgrep -f here previously matched itself and spun forever, idling the GPU.
set -u
cd "$(dirname "$0")"
LOG=~/research/logs

running() { pgrep -fc "python -u s[57]_" 2>/dev/null; true; }

waitslot() {
  while [ "$(running)" -ge 3 ]; do sleep 30; done
}

launch() {   # launch <logname> <command...>
  local name=$1; shift
  waitslot
  echo "$(date +%H:%M:%S) launching $name ($(running) running)"
  setsid nohup "$@" > "$LOG/$name.log" 2>&1 < /dev/null &
  sleep 45          # let it claim its slot before the next slot check
}

echo "=== queue start $(date) ==="

# Table 1's learnable-delay rows are means over three seeds from an implementation whose delays could
# not move. Seeds 2 and 3 of the corrected learner make the replacement comparable to the rest.
launch q_dcls_s2 env S=2 bash run_dcls002_seeds.sh
launch q_dcls_s3 env S=3 bash run_dcls002_seeds.sh

echo "=== queue submitted $(date) ==="
