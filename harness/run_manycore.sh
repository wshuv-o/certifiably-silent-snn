#!/usr/bin/env bash
# CORE-SCALING-002 (pre-registered in N3_SCALEUP_PLAN.md): the SPEED-DIAG-001 protocol extended past 32 cores,
# for a machine with >= 128 physical cores. Identical measurement, idle guard and repeats; only the core list
# grows, and core counts above the machine's physical core count are skipped rather than oversubscribed.
#
# Usage on the rented machine:  SRC=<bundle>/data bash run_manycore.sh | tee manycore.log
#
# --- original SPEED-DIAG-001 header follows ---
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
SRC=${SRC:-~/research/aws_bundle/data}
WORK=${WORK:-~/research/manycore}
BIN=$WORK/s6_engine_delays
REPEATS=${REPEATS:-3}

if [ "${SKIP_IDLE_GUARD:-0}" != 1 ]; then   # set only for a local pipeline dry-run, never for timings
busy=$(pgrep -fc "python -u s[57]_" 2>/dev/null; true)
if [ "$busy" -gt 0 ]; then
  echo "REFUSING: $busy training process(es) still on the CPU. A loaded machine is the confound."
  exit 1
fi
load=$(awk '{print int($1+0.5)}' /proc/loadavg)
if [ "$load" -gt 2 ]; then
  echo "REFUSING: load average $load. Wait for the machine to settle."
  exit 1
fi
fi

mkdir -p "$WORK"
cp -r "$SRC"/. "$WORK/"          # never mutate the bundle: dims.txt is rewritten per core count

echo "=== host: $(nproc) threads, load $(cut -d' ' -f1-3 /proc/loadavg) ==="
if [ -x "$SRC/../s6_engine_delays.static" ]; then
  # prebuilt static binary (x86-64-v3), shipped in the bundle so the host needs no C++20 compiler
  cp "$SRC/../s6_engine_delays.static" "$BIN"; rc=0
else
  g++ -O3 -march=native -std=c++20 -pthread s6_engine_delays.cpp -o "$BIN"
  rc=$?
fi
sz=$(stat -c%s "$BIN" 2>/dev/null || echo 0)
echo "compile rc=$rc, binary $sz bytes"
# a previous g++ here produced a 0-byte binary and still reported success through a pipeline
[ "$rc" -ne 0 ] && { echo "FATAL: compile failed"; exit 1; }
[ "$sz" -lt 10000 ] && { echo "FATAL: binary too small, refusing to measure"; exit 1; }

read -r P0 C0 T N REST < "$WORK/cert_dims.txt"
H=$((P0 * C0))
echo "=== model H=$H, N=$N samples, T=$T, L=0, $REPEATS repeats ==="
echo

PHYS=$(lscpu -p=SOCKET,CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l); PHYS=${PHYS:-$(nproc)}
echo "=== physical cores: $PHYS ==="
for CORES in ${CORE_LIST:-4 8 16 32 64 128 256}; do
  [ "$CORES" -gt "$PHYS" ] && { echo "skip CORES=$CORES (> $PHYS physical cores; would oversubscribe)"; continue; }
  [ $((H % CORES)) -ne 0 ] && { echo "skip CORES=$CORES"; continue; }
  C=$((H / CORES))
  for m in cert ctrl; do
    awk -v p="$CORES" -v c="$C" '{$1=p; $2=c; print}' "$SRC/${m}_dims.txt" > "$WORK/${m}_dims.txt"
  done
  for r in $(seq 1 "$REPEATS"); do
    # cert model under both certificates (CERT_ADAPT=0 plain, 1 adaptation-aware), then the control
    for ad in ${ADAPT_LIST:-0 1}; do
      CERT_ADAPT=$ad "$BIN" "$WORK" cert 2 2>&1 | grep -a "^RESULT" | sed "s/^/CORES=$CORES rep=$r ADAPT=$ad /"
    done
    CERT_ADAPT=0 "$BIN" "$WORK" ctrl 2 2>&1 | grep -a "^RESULT" | sed "s/^/CORES=$CORES rep=$r ADAPT=0 /"
  done
  echo
done
echo "=== CORE-SCALING-002 COMPLETE $(date) ==="
