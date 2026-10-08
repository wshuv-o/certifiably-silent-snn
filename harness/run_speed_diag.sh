#!/usr/bin/env bash
# SPEED-DIAG-001 (pre-registered in N3_SCALEUP_PLAN.md): is the 32-core speed-up real or saturation?
#
# This machine has exactly 32 threads, so running 32 cores saturates it and any gain may come from
# the handshake baseline being descheduled rather than from certificates removing waiting. The engine
# separates blocked time from compute time, which distinguishes the two:
#
#   real        -> comp_ms flat across core counts, handshake wait_ms grows smoothly
#   saturation  -> handshake wait_ms jumps at 32 while smooth to 16, and comp_ms inflates for BOTH
#                  modes at 32 because threads lose the CPU mid-step
#
# Refuses to run if anything else is on the CPU: a loaded machine is the confound under test, and is
# how the first two attempts to measure this engine were lost.
set -u
cd "$(dirname "$0")"
SRC=~/research/aws_bundle/data
WORK=~/research/speeddiag
BIN=$WORK/s6_engine_delays
REPEATS=${REPEATS:-3}

busy=$(pgrep -fc "python -u s[57]_" 2>/dev/null || echo 0)
if [ "$busy" -gt 0 ]; then
  echo "REFUSING: $busy training process(es) still on the CPU. A loaded machine is the confound."
  exit 1
fi
load=$(awk '{print int($1+0.5)}' /proc/loadavg)
if [ "$load" -gt 2 ]; then
  echo "REFUSING: load average $load. Wait for the machine to settle."
  exit 1
fi

mkdir -p "$WORK"
cp -r "$SRC"/. "$WORK/"          # never mutate the bundle: dims.txt is rewritten per core count

echo "=== host: $(nproc) threads, load $(cut -d' ' -f1-3 /proc/loadavg) ==="
g++ -O3 -march=native -std=c++20 -pthread s6_engine_delays.cpp -o "$BIN"
rc=$?
sz=$(stat -c%s "$BIN" 2>/dev/null || echo 0)
echo "compile rc=$rc, binary $sz bytes"
# a previous g++ here produced a 0-byte binary and still reported success through a pipeline
[ "$rc" -ne 0 ] && { echo "FATAL: compile failed"; exit 1; }
[ "$sz" -lt 10000 ] && { echo "FATAL: binary too small, refusing to measure"; exit 1; }

read -r P0 C0 T N REST < "$WORK/cert_dims.txt"
H=$((P0 * C0))
echo "=== model H=$H, N=$N samples, T=$T, L=0, $REPEATS repeats ==="
echo

for CORES in 4 8 16 32; do
  [ $((H % CORES)) -ne 0 ] && { echo "skip CORES=$CORES"; continue; }
  C=$((H / CORES))
  for m in cert ctrl; do
    awk -v p="$CORES" -v c="$C" '{$1=p; $2=c; print}' "$SRC/${m}_dims.txt" > "$WORK/${m}_dims.txt"
  done
  for m in cert ctrl; do
    for r in $(seq 1 "$REPEATS"); do
      "$BIN" "$WORK" "$m" 2 2>&1 | grep -a "^RESULT" | sed "s/^/CORES=$CORES rep=$r /"
    done
  done
  echo
done
echo "=== SPEED-DIAG-001 COMPLETE $(date) ==="
