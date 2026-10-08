#!/usr/bin/env bash
# Many-core run of the shared-memory engine, for a machine with far more threads than this project
# has had access to. Two questions:
#
#   1. Does the certificate speed-up keep growing past 32 cores? Measured so far: 0.93, 0.90, 0.79,
#      1.44 at 4, 8, 16, 32.
#   2. Is the 1.44x at 32 cores real? The development machine has exactly 32 threads, so running 32
#      cores saturates it and the handshake baseline may be suffering rather than certificates
#      helping. On a 128-thread box 32 cores no longer saturates, so the number is clean.
#
# Only dims.txt depends on the partition -- W*.bin and I.bin are identical for every core count --
# so the export is shipped once and dims.txt is rewritten per core count.
set -u
cd "$(dirname "$0")"
DATA=./data
BIN=./s6_engine_delays

echo "=== host ==="
nproc; grep -m1 "model name" /proc/cpuinfo | cut -d: -f2-; free -g | awk '/Mem/{print $2" GB RAM"}'
uptime | sed 's/.*up/up/'

echo "=== build ==="
g++ -O3 -march=native -std=c++20 -pthread s6_engine_delays.cpp -o "$BIN" || exit 1
sz=$(stat -c%s "$BIN"); echo "binary $sz bytes"
[ "$sz" -lt 10000 ] && { echo "FATAL: binary too small"; exit 1; }

# the shipped dims.txt, whose fields are: P C T N BETA THETA RHO GAMMA ND delays...
read -r P0 C0 T N REST < "$DATA/cert_dims.txt"
H=$((P0 * C0))
echo "=== model H=$H, $N samples, T=$T ==="

for CORES in 4 8 16 32 64 128; do
  if [ $((H % CORES)) -ne 0 ]; then echo "skip CORES=$CORES (H=$H not divisible)"; continue; fi
  C=$((H / CORES))
  for m in cert ctrl; do
    # rewrite only the partition fields, keep every physical constant byte-identical
    awk -v p="$CORES" -v c="$C" '{$1=p; $2=c; print}' "$DATA/${m}_dims.txt" > "$DATA/${m}_dims.new"
    mv "$DATA/${m}_dims.new" "$DATA/${m}_dims.txt"
  done
  echo "########## CORES=$CORES  C=$C  $(date +%H:%M:%S) ##########"
  for m in cert ctrl; do
    echo "---- $m ----"
    "$BIN" "$DATA" "$m" 2
  done
done
echo "=== COMPLETE $(date +%H:%M:%S) ==="
