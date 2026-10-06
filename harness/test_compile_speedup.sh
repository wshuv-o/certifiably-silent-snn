#!/usr/bin/env bash
# Is torch.compile(mode="reduce-overhead") a safe speed-up for this launch-bound model?
# It fuses kernels and uses CUDA graphs, which is exactly what a 100-timestep loop over small matmuls
# needs. RISK: fusion can reassociate floating-point ops, so results may shift slightly. This script
# therefore VALIDATES before adopting: it re-runs an already-measured config and compares.
# Waits for the GPU to be free so the timing is meaningful.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=~/research/logs; M=~/research/models
echo "=== waiting for GPU to free ==="
while pgrep -f "s4_improve.py" > /dev/null; do sleep 20; done
echo "=== GPU free $(date +%H:%M:%S); baseline (no compile), 10 epochs, seed 1 ==="
/usr/bin/time -f "WALL %e s" env TAG=cmp_off SEED=1 EPOCHS=10 TAUM=2.0 python -u s4_improve.py > "$LOG/cmp_off.log" 2>&1
grep -h "^RESULT" "$LOG/cmp_off.log"; grep "WALL" "$LOG/cmp_off.log" || tail -2 "$LOG/cmp_off.log"
echo "=== with torch.compile, 10 epochs, same seed ==="
/usr/bin/time -f "WALL %e s" env TAG=cmp_on SEED=1 EPOCHS=10 TAUM=2.0 COMPILE=1 python -u s4_improve.py > "$LOG/cmp_on.log" 2>&1
grep -h "^RESULT" "$LOG/cmp_on.log"; grep "WALL" "$LOG/cmp_on.log" || tail -2 "$LOG/cmp_on.log"
echo "=== compare acc_val / core_cert / R_mean above: adopt ONLY if they match within noise ==="
