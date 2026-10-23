#!/usr/bin/env bash
# SPEED-005 probe (pre-registered addendum): handshake / certificate (CERT_FAST, adaptive) / oracle upper bound,
# on the dense R1 network and its synthetic local version, 4-64 physical cores, idle machine.
set -u
cd "$(dirname "$0")"
SRC=${SRC:-./data}; WORK=${WORK:-/tmp/probe}; REPEATS=${REPEATS:-2}
load=$(awk '{print int($1+0.5)}' /proc/loadavg)
[ "${SKIP_IDLE_GUARD:-0}" != 1 ] && [ "$load" -gt 2 ] && { echo "REFUSING: load $load"; exit 1; }
mkdir -p "$WORK"; cp -r "$SRC"/. "$WORK/"; BIN=$WORK/eng; cp ./s6_engine_delays.static "$BIN"
python3 make_local_probe.py "$WORK" || exit 1
for f in cert certloc; do cp "$WORK/${f}_dims.txt" "$WORK/${f}_dims.orig"; done
PHYS=$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l); echo "=== physical cores: $PHYS ==="
read -r P0 C0 REST < "$WORK/cert_dims.orig"; H=$((P0 * C0))
for CORES in ${CORE_LIST:-4 8 16 32 64}; do
  [ "$CORES" -gt "$PHYS" ] && { echo "skip CORES=$CORES"; continue; }
  C=$((H / CORES))
  for f in cert certloc; do awk -v p="$CORES" -v c="$C" '{$1=p; $2=c; print}' "$WORK/${f}_dims.orig" > "$WORK/${f}_dims.txt"; done
  for r in $(seq 1 "$REPEATS"); do
    for m in cert certloc; do
      CERT_FAST=1 CERT_ADAPT=1 "$BIN" "$WORK" "$m" 3 2>&1 | grep -a "^RESULT\|inter-core" | sed "s/^/CORES=$CORES rep=$r /"
    done
  done
done
echo "=== SPEED-005 PROBE COMPLETE $(date) ==="
