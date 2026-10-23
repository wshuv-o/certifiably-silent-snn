# Execution across two processes over TCP (supplement, TCP table)

Engine `harness/s8_engine_net.cpp`, 32 cores, 2 ranks, 100 samples, idle machine, 3 repeats per point;
spike trains identical to the reference in all 36 runs. Median ms per sample; `blocked` is time the main loop
waits on the peer; `compute` = median - blocked.

| L (us) | mode | median | min-max | blocked | compute |
|---|---|---|---|---|---|
| 200 | handshake | 17.22 | 16.79-17.25 | 3.70 | 13.56 |
| 200 | certificate | 19.73 | 19.52-20.06 | 1.12 | 18.40 |
| 400 | handshake | 26.71 | 26.44-27.28 | 13.35 | 13.28 |
| 400 | certificate | 26.03 | 25.64-27.11 | 7.43 | 18.61 |
| 800 | handshake | 47.09 | 46.17-47.32 | 32.53 | 14.56 |
| 800 | certificate | 41.67 | 41.05-42.61 | 22.85 | 18.82 |
| 1600 | handshake | 86.50 | 86.46-86.55 | 73.15 | 13.32 |
| 1600 | certificate | 73.44 | 73.02-73.77 | 54.85 | 18.59 |

The certificate adds 5.06 ms of computation per sample and removes waiting of 2.57, 5.92, 9.68 and 18.30 ms
at L = 200, 400, 800 and 1600 us; the two balance at L = 348 us. Speed-up 1.13x at 800 us, 1.18x at 1600 us.
