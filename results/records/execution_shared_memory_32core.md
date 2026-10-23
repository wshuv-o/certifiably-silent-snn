# Shared-memory execution on a 32-thread workstation (figure 2f)

Engine `harness/s6_engine_delays.cpp`, driver `harness/run_speed_diag.sh`; 240 runs, 3 repeats per point,
idle machine (1-minute load below 0.6), spike trains identical to the reference in every run.
Speed-up = handshake time / certificate time.

| cores | certified coverage | L=0 | L=5 | L=20 | L=100 | L=500 (us) |
|---|---|---|---|---|---|---|
| 4 | 46.5% | 0.932 | 0.938 | 0.946 | 1.061 | 1.113 |
| 8 | 49.6% | 0.873 | 0.836 | 0.883 | 1.067 | 1.117 |
| 16 | 56.3% | 0.808 | 0.805 | 0.809 | 1.050 | 1.114 |
| 32 | 71.4% | 0.730 | 0.726 | 0.721 | 1.067 | 1.144 |
| 32, unconstrained control | 5.5% | 0.763 | 0.822 | 0.838 | 0.968 | 0.963 |
