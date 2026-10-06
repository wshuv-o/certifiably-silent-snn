#!/usr/bin/env bash
# SWEEP-002 stage 2: fine-tuning / distillation configs (need S0). Seed 1.
# RTX 5080 (16 GB): 6 parallel queues, one config each (~1-1.5 GB per run).
# Previously 2 queues of 3 on the RTX 2060 (6 GB). Parallelism only: identical
# seeds, configs and selection rule, so results are unaffected.
# Logging note (2026-10-06): see run_sw2_stage1.sh -- full logs, no grep pipe.
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
M=~/research/models; LOG=~/research/logs; mkdir -p "$M" "$LOG"
T0=$M/sw2_S0_ctrl_s1.pt
if [ ! -f "$T0" ]; then echo "*** ABORT: teacher/init checkpoint $T0 missing; run stage 1 first ***"; exit 1; fi
FT="INIT=$T0 EPOCHS=20 LR=5e-4 RAMP=1"
run() { tag=$1; shift; env SEED=1 TAG=$tag "$@" SAVE=$M/sw2_${tag}_s1.pt python -u s4_improve.py > "$LOG/$tag.log" 2>&1; }
( run S2_ft_cert03         $FT CERT_LAMBDA=0.3 ) &
( run S3_ft_cert03_kd      $FT CERT_LAMBDA=0.3 TEACHER=$T0 ) &
( run S4_ft_cert10_kd      $FT CERT_LAMBDA=1.0 TEACHER=$T0 ) &
( run S6_ft_hub4_kd        $FT NHUB=4 CERT_LAMBDA=0.3 TEACHER=$T0 ) &
( run S7_ft_excap_kd       $FT ALT=clamp TEACHER=$T0 ) &
( run S5_scratch_cert03_kd     CERT_LAMBDA=0.3 TEACHER=$T0 ) &
wait
fail=0
for t in S2_ft_cert03 S3_ft_cert03_kd S4_ft_cert10_kd S5_scratch_cert03_kd S6_ft_hub4_kd S7_ft_excap_kd; do
  if grep -q "^RESULT" "$LOG/$t.log"; then grep -h "^RESULT" "$LOG/$t.log"; else
    echo "*** FAILED: $t ***"; tail -25 "$LOG/$t.log"; fail=1; fi
done
exit $fail
