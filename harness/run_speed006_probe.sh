#!/usr/bin/env bash
# SPEED-006 probe: handshake / certificate (CERT_FAST, adaptive) / oracle for each exported model, 4-64 cores.
set -u
cd "$(dirname "$0")"
SRC=${SRC:-./data}; WORK=${WORK:-/tmp/probe6}; REPEATS=${REPEATS:-2}; MODELS=${MODELS:-dense lk4 lk8}
load=$(awk '{print int($1+0.5)}' /proc/loadavg)
[ "${SKIP_IDLE_GUARD:-0}" != 1 ] && [ "$load" -gt 2 ] && { echo "REFUSING: load $load"; exit 1; }
mkdir -p "$WORK"; cp -r "$SRC"/. "$WORK/"; BIN=$WORK/eng; cp ./s6_engine_delays.static "$BIN"
for m in $MODELS; do cp "$WORK/${m}_dims.txt" "$WORK/${m}_dims.orig"; done
PHYS=$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l); echo "=== physical cores: $PHYS ==="
for CORES in ${CORE_LIST:-4 8 16 32 64}; do
  [ "$CORES" -gt "$PHYS" ] && { echo "skip CORES=$CORES"; continue; }
  for m in $MODELS; do
    read -r P0 C0 REST < "$WORK/${m}_dims.orig"; H=$((P0 * C0)); C=$((H / CORES))
    awk -v p="$CORES" -v c="$C" '{$1=p; $2=c; print}' "$WORK/${m}_dims.orig" > "$WORK/${m}_dims.txt"
  done
  for r in $(seq 1 "$REPEATS"); do
    for m in $MODELS; do
      CERT_FAST=1 CERT_ADAPT=1 "$BIN" "$WORK" "$m" 3 2>&1 | grep -a "^RESULT\|inter-core\|R_short" | sed "s/^/CORES=$CORES rep=$r /"
    done
  done
done
echo "=== SPEED-006 PROBE COMPLETE $(date) ==="
