# Experiment Ledger (Layer 3)

One entry per experiment. Change as few variables as possible per diagnostic experiment.

---

## EXP-001-pilot — DS vs SS crossover (2026-10-04) — DISCARDED as evidence

- **Hypothesis:** P1. The sparsity crossover is s* ≈ χ/κ, with χ = c_e/c_m.
- **Config:** N = 2^20, F ∈ {8, 32, 64}, window ∈ {256 (local), random}, 12 activity levels, 7 reps, DS then SS run sequentially. Machine: i7-10750H (6 threads) and RTX 2060 Laptop, under WSL2.
- **Result:** verification PASS. But GPU DS time varied up to 5× (coefficient of variation up to 5.2) with no trend in s.
- **Diagnosis:** NVML showed the SM clock at 300–1860 MHz and temperature up to 87 °C. The laptop GPU drops clock states between short kernels.
- **Conclusion:** this is a protocol defect, not an algorithmic effect. Files: `results/exp001.csv` (the pilot; rename was blocked by a file lock).
- **Next action:** paired, interleaved protocol (v2).

Earlier diagnostic in the same session: under WSL, CPU timing with 12 threads and passive OpenMP gave ~20 ms stalls on 0.3 ms steps. Fixed by using 6 threads with `OMP_WAIT_POLICY=active` (median ≈ 1.3 × minimum).

## EXP-001 v2 — DS vs SS crossover, paired protocol (2026-10-04)

- **Hypothesis:** P1 (as above). Open-loop, uniform active sets (κ ≈ 1).
- **Change from pilot:**
  - DS and SS interleaved every rep, with alternating order.
  - GPU warm-up (0.3 s) before each cell.
  - 15 reps.
  - SM clock logged per rep.
  - Crossover taken from the median of paired SS/DS ratios.
- **Verification:** PASS. All 4 platform×mode combinations matched the NumPy reference at s ∈ {1e-3, 0.1, 1}.
- **Stability:** coefficient of variation of DS time over s: GPU 0.03–0.13, CPU 0.05–0.27. GPU clock during timed reps: median 1185 MHz, p5 1005, min 630. Median relative IQR 0.086; the worst cell is 7.4, so some outlier cells remain.
- **Results** (s* measured; predicted from the fitted SS line in parentheses):

| | F=8 | F=32 | F=64 | χ = c_e/c_m |
|---|---|---|---|---|
| CPU random | 0.194 (0.193) | 0.140 (0.161) | 0.140 (0.150) | 0.142 |
| CPU local | 0.152 (0.152) | 0.093 (0.110) | 0.096 (0.101) | 0.094 |
| GPU random | 0.078 (0.088) | 0.077 (0.079) | 0.078 (0.081) | 0.080 |
| GPU local | 0.566 (0.406) | 0.220 (0.230) | 0.182 (0.173) | 0.148 |

- **Observations:**
  1. For F ≥ 32, measured s* ≈ χ within about 5% in 3 of 4 platform×locality cases (CPU random, CPU local, GPU random). GPU local is 0.18–0.22 against χ = 0.148.
  2. At F = 8, s* is higher than χ, as expected: the approximation needs F ≫ 1, and per-unit costs matter at low F.
  3. **Locality moves the crossover in opposite directions on the two platforms.**
     - On CPU, local destinations *lower* s*: dense mode gains more from cache locality.
     - On GPU, local destinations *raise* s* (0.08 → 0.18–0.57): coalesced atomics make sparse push much cheaper.
     - So the same workload property shifts the regime boundary differently depending on hardware. This supports the hardware-relative framing over "sparsity alone".
  4. Sparse pays off only below ~8–20% activity on these platforms (κ ≈ 1), well below "anything sparse helps".
- **Interrogation (do not over-claim):**
  - Agreement between s* and χ is partly expected. χ comes from the same runs, and the cost model is linear. It shows the linear model describes both platforms; it does **not** yet test transfer. A real P1 test predicts s* from *independent* microbenchmarks, or predicts a platform/configuration not used in fitting.
  - Single graph seed, single active-set seed, κ ≈ 1 only, one laptop, SS vs DS only (no asynchrony yet).
  - Our SS kernels are not yet checked against gapbs/Gunrock (strawman guard pending). A weak SS kernel would push s* down.
  - Some timing outliers remain (max IQR 7.4); the paired ratio mitigates but does not remove them.
- **Files:** `results/exp001v2.csv`, `_meta.json`, `_summary.json`. Code: `harness/` (graphgen, cpu_kernels, gpu_kernels, bench_sparsity, analyze_sparsity).
- **Next actions:**
  1. Strawman guard: compare our SS/DS against gapbs (CPU) on the same graphs. → done, see GUARD-001.
  2. Out-of-sample P1 test: fit constants on F ∈ {8, 64}, predict s* at F = 32 and at a new N. Add seeds.
  3. Vary κ (clustered active sets) to test the 1/κ term.
  4. Then add the AE mode (asynchrony component).

## GUARD-001 — strawman guard vs gapbs, CPU (2026-10-04)

- **Same graph:** N = 2^20, F = 32, random destinations; gapbs commit 2972aeb; 6 threads with active OpenMP wait, for both.
- **Raw per-edge cost:**
  - Dense: ours 1.10–1.16 ns vs gapbs PageRank 0.57–0.65 ns, **1.8–1.9×**.
  - Sparse: ours (Numba) 8.1–11.5 ns vs gapbs BFS top-down 5.1–5.9 ns, **1.4–2.3×**.
  - A first parse used the wrong frontier (gapbs logs the *output* frontier, bfs.cc:171) and showed a spurious 30–40×; corrected.
- **Equal-work check** (our C++ kernels doing gapbs's per-edge work):
  - DS without weights: 0.74 ns/edge, **1.1–1.3× of PageRank → PASS.**
  - SS without accumulation: 9.1 ns/edge at a ~31k frontier, **~1.6× of BFS top-down → near-pass.** The residual is the extra next-frontier pass (touched-list reset and threshold check), which a BFS top-down step does not do.
- **Alternative tried:** a gapbs-style single-pass C++ SS with atomic float add (`harness/cpu_ref.cpp`). Verified, but **slower** than Numba bucketed SS at low s (15–18 vs 11 ns/edge) and equal at s = 0.1. **Decision:** keep Numba SS as the CPU SS implementation.
- **Known weakness:** CPU SS may be up to ~1.6× slower than the best achievable. Since s* scales ~linearly with SS per-active cost, the CPU s* values may be underestimated by up to ~1.6×. This biases *against* sparse execution (conservative for the hypothesis direction), but it must be stated.
- **Observation:** absolute CPU constants drift with thermal state. DS measured alone gives 1.10 ns/edge; inside the EXP-001 v2 interleaved run, 1.85 ns/edge. **Paired ratios and crossovers are primary; absolute constants are condition-dependent.**
- **Pending:** GPU guard against Gunrock or Atos.

## EXP-002 — out-of-sample prediction of s* (PRE-REGISTERED 2026-10-04, before running)

- **Fitted only on** EXP-001 v2 constants (N = 2^20, F ∈ {8, 32, 64}, seed 0).
- **Unseen cases:** (N = 2^20, F = 16), (N = 2^20, F = 48), (N = 2^19, F = 32), with new graph and activity seeds (exp002a: seed 1; exp002b: seed 2).
- **Assumption:** the dense fixed term scales ∝ N.
- **Predictions** (`results/exp002_predictions.json`):

| platform, window | F=16, N=2^20 | F=48, N=2^20 | F=32, N=2^19 |
|---|---|---|---|
| CPU random | 0.172 | 0.154 | 0.158 |
| CPU local | 0.125 | 0.104 | 0.109 |
| GPU random | 0.082 | 0.081 | 0.079 |
| GPU local | 0.282 | 0.190 | 0.181 |

- **Success criterion** (fixed now): median |relative error| ≤ 0.20 and no case off by more than 0.50. Otherwise the linear hardware-relative cost model is **not** supported out of sample, and we diagnose which term fails.

## EXP-003 — spatial imbalance κ (PRE-REGISTERED 2026-10-04, before running)

- **Question:** does load imbalance shift the sparsity crossover as the model's κ term says (s* ≈ χ/κ)?
- **Manipulation:** the active set is drawn from one contiguous region of ρN units, with ρ ∈ {1, 0.5, 0.25, 0.167, 0.1}.
- **Measured κ:** max/mean messages per destination owner (6 contiguous blocks, matching the CPU SS kernel), logged per cell.
- **Config:** N = 2^20, F = 32, windows {256 (local), random}, seed 3, same paired protocol as EXP-001 v2.
- **Predictions (fixed now):**
  1. **CPU local:** κ rises toward min(6, 1/ρ), and s* decreases monotonically as ρ falls. Magnitude: s*(ρ)/s*(1) lies between 1/κ (full imbalance penalty) and 1. Only the owner-apply pass is imbalanced in our SS kernel, so a partial effect is expected.
  2. **CPU random:** messages scatter to all owners, so κ ≈ 1 for every ρ, and s* changes by less than 15%. (This is the control: same spatial clustering of *senders*, but no imbalance at *receivers*.)
  3. **GPU:** exploratory, no directional claim. GPU scheduling is dynamic, so imbalance across our 6 notional owners has no direct meaning; possible effects come from atomic contention or locality.
- **Falsification:** if CPU local s* does not decrease with κ, or CPU random s* moves by more than 15%, the κ term as modelled is wrong for this platform.

### EXP-002 — RESULT: pre-registered criterion FAILED

- **Verification:** PASS (both runs).
- **Accuracy:** median |relative error| 0.228 (criterion ≤ 0.20); max 1.25 (criterion ≤ 0.50). Table in `results/exp002_comparison.csv`.
- **By regime:**
  - **CPU (6 cases):** all measured s* are 12–24% *below* prediction. The sign is consistent, so the F and N dependence is right but the level is biased. Consistent with the thermal-state dependence of absolute CPU constants (GUARD-001).
  - **GPU random, N = 2^20 (F = 16, 48):** errors −10% and −7%. Supported.
  - **GPU random, N = 2^19:** −54%.
  - **GPU local:** +57% to +125%. The in-sample fit there was already unstable (negative b).
- **Diagnosis (EXP-001 v2 vs EXP-002b, F = 32, at s ≥ 0.1):**
  - GPU random, going from N = 2^20 to 2^19: DS cost per edge 0.249 → 0.083 ns (3.0× cheaper); SS cost per message 2.99 → 1.84 ns (1.6× cheaper).
  - So χ_operating = 0.083 → 0.045, and measured s* = 0.078 → 0.036. **s* still tracks the χ measured at the operating point.**
  - Cause: the state vector is 4 MB at 2^20 and 2 MB at 2^19, against 3 MB of L2 on the RTX 2060. Dense gathers move into L2 and gain far more than atomic pushes do.
  - CPU χ_operating: 0.183 → 0.148 (random), 0.126 → 0.128 (local).
- **Conclusion:** the claim that s* is governed by the dense/event cost ratio survives. The claim that this ratio is a **portable hardware constant** is **refuted**: χ depends on working set relative to cache capacity, so it is χ(H, M) and not χ(H). This makes state size M a first-class workload dimension and requires a memory-hierarchy term in the cost model.
  - It also explains GUARD-001's drift partly; the thermal hypothesis is not ruled out.
- **New hypothesis H-cache (to pre-register as EXP-004):** on the GPU, s* is roughly constant while the state vector exceeds L2 (N ≥ 2^20), and drops once it fits (N ≤ 2^19). Test with N ∈ {2^17, 2^18, 2^19, 2^20, 2^21, 2^22}, F = 32, random destinations.
- **Code hygiene:** `analyze_sparsity.py` ran its analysis on import, which reprinted pilot tables into the EXP-002 log. The crossing rule now lives in `harness/common.py`.

## EXP-004 — working set vs cache (H-cache) (PRE-REGISTERED 2026-10-04, before running)

- **Config:** F = 32, random destinations (window 0), N ∈ {2^17, 2^18, 2^19, 2^20, 2^21, 2^22}, seed 4, paired protocol.
- **Cache sizes:** float32 state/x vector = 4N bytes; RTX 2060 L2 = 3 MB (crossed between 2^19 = 2 MB and 2^20 = 4 MB); i7-10750H L3 = 12 MB (crossed between 2^21 = 8 MB and 2^22 = 16 MB).
- **Predictions (fixed now):**
  1. **GPU:** s* for N ≤ 2^19 is below 0.05. s* for N ≥ 2^20 lies in 0.06–0.10 and is roughly flat (within ±25% of its mean). The transition sits between 2^19 and 2^20.
  2. **CPU:** s*(2^22) exceeds s*(2^20) by at least 15% (the dense gather leaves L3, so dense becomes relatively costlier). Weaker prediction, because CPU constants also drift thermally.
  3. **Both:** measured s* stays within 35% of the operating-point ratio χ_op = (DS ns/edge) / (SS ns/message at s ≥ 0.1) at each N. This is the "s* tracks the local cost ratio" claim, refined.
- **Falsification:** no GPU step change between 2^19 and 2^20 refutes H-cache. Large deviations of s* from χ_op refute the refined ratio claim.

### EXP-003 — RESULT

- **Verification:** PASS. Table: `results/exp003_summary.csv`.

| | ρ=1 | 0.5 | 0.25 | 0.167 | 0.1 |
|---|---|---|---|---|---|
| CPU local: κ / s* | 1.01 / 0.088 | 2.00 / 0.080 | 3.62 / 0.078 | 4.60 / 0.080 | 4.74 / 0.076 |
| CPU random: κ / s* | 1.00 / 0.126 | 1.00 / 0.141 | 1.00 / 0.142 | 1.00 / 0.145 | 1.00 / 0.142 |
| GPU random: s* | 0.076 | 0.072 | 0.077 | 0.075 | 0.075 |
| GPU local: κ / s* | 1.01 / 0.358 | 2.01 / 0.310 | 3.32 / 0.263 | 3.33 / 0.262 | 3.33 / 0.260 |

- **Prediction 1 (CPU local):**
  - *Direction:* supported, weakly. s* falls 9–13% as κ goes to ~4.7. It is not strictly monotone (0.078 → 0.080), but the step is within noise.
  - *Magnitude:* the model's 1/κ penalty predicts up to −79%; observed is −13%. The pre-registered bound "between 1/κ and 1" was satisfied, but that bound was too wide to be informative (an honest note on a weak pre-registration).
  - **The κ term as modelled (T_SS ≈ κ·W/P) is refuted in magnitude** for this kernel. Only the owner-apply pass is imbalanced, and it is a minor share of SS time.
- **Prediction 2 (CPU random control):** κ ≈ 1 as predicted. But s* rose 12–15% (0.126 → 0.141–0.145), right at the 15% limit (1.147). Cause: clustered senders read contiguous `dst`/`w` ranges, so SS gets cheaper through **sender-side locality**, not through imbalance.
- **Prediction 3 (GPU, exploratory):**
  - Random destinations: no effect (±5%).
  - Local destinations: clustering *lowers* s* by 27% (0.358 → 0.260). Likely atomic contention: concentrated senders plus a local window put many concurrent atomics on a small address range.
- **Conclusion:**
  - Spatial concentration of activity matters, but mostly **not through load imbalance**. It acts through locality (CPU: helps sparse) and contention (GPU: hurts sparse), with **opposite signs on the two platforms**.
  - "κ" as a single workload dimension conflates imbalance, locality and contention. The confounding between L and κ was anticipated in Phase 4 §3.
  - This is further evidence that regime boundaries are hardware-relative. It also shows the analytical model needs a contention term for GPU atomics and a sender-locality term.
- **Model revisions queued:**
  - Replace κ·W/P with: imbalance applied only to the imbalanced phase, plus a locality-dependent c_m and a contention-dependent atomic cost.
  - Add χ(H, M) from EXP-002.

## EXP-005 — asynchrony component on independent code (Galois) (PRE-REGISTERED 2026-10-04, before running)

- **Question:** P2. Does the asynchrony gain G_async = T_sync / T_async grow when per-round work is small relative to barrier cost (many rounds, small frontiers)? And does it vanish when rounds are few and frontiers large?
- **Code:** Galois b67f942. BFS `Sync` vs `Async`; SSSP `deltaStepBarrier` vs `deltaStep`. 6 threads, 7 paired trials with alternating order, plus 1 warm-up trial.
- **Graphs** (depth controlled through the locality window; recorded: BFS rounds and mean frontier fraction from vertex 0):
  - Synthetic, N = 2^20, F = 8: window 16, 256, 4096 and random. A smaller window gives more rounds and smaller frontiers.
  - SNAP roadNet-CA (real, high diameter).
- **Predictions (fixed now):**
  1. **BFS:** G_async increases monotonically with BFS rounds across the 4 synthetic graphs (w16 > w256 > w4096 > random).
  2. **Random graph** (few rounds, frontiers up to ~10–50% of N): G_async ≤ 1.1 for both kernels. Async gains nothing or loses, through redundant work (ω > 1).
  3. **roadNet-CA:** G_async exceeds that of the random synthetic graph, for both kernels.
  4. **SSSP:** same ordering as prediction 1, but a weaker claim (delta-stepping buckets already reduce barrier count).
- **Caveat recorded in advance:** Galois `Async` and `Sync` differ in more than barrier removal (worklist type, scheduling). So G_async here measures "Galois's async design vs its sync design", not pure barrier cost. Our own AE kernel (to be built) is needed for the clean decomposition.

### EXP-004 — RESULT

- **Process note:** the first launch ran every size at N = 1, because WSL's outer shell expanded `$k` before the inner bash loop ran (a scripting error; results discarded, and `exp004_n.csv` is excluded by the analysis glob). N = 2^22 then died from memory pressure in the NumPy reference check, which was fixed by chunking `reference_step`. Final runs are via `harness/run_exp004.sh`. **Verification:** PASS at all 6 sizes.

| log2N | state MB | CPU DS ns/edge | CPU SS ns/msg | CPU χ_op | CPU s* | GPU DS ns/edge | GPU SS ns/msg | GPU χ_op | GPU s* |
|---|---|---|---|---|---|---|---|---|---|
| 17 | 0.5 | 0.90 | 6.24 | 0.145 | 0.121 | 0.117 | 0.161 | 0.725 | ∞ (SS always wins) |
| 18 | 1 | 0.93 | 6.43 | 0.145 | 0.122 | 0.092 | 0.319 | 0.287 | 0.270 |
| 19 | 2 | 0.98 | 6.48 | 0.151 | 0.126 | 0.080 | 1.582 | 0.051 | 0.040 |
| 20 | 4 | 1.06 | 6.85 | 0.155 | 0.127 | 0.231 | 2.602 | 0.089 | 0.079 |
| 21 | 8 | 1.56 | 7.44 | 0.209 | 0.172 | 0.433 | 3.182 | 0.136 | 0.126 |
| 22 | 16 | 2.48 | 10.87 | 0.228 | 0.241 | 0.641 | 3.474 | 0.185 | 0.177 |

- **Prediction 1 (GPU step at L2, flat above): FAILED.**
  - s* is non-monotonic in N: ∞ → 0.27 → 0.04 → 0.08 → 0.13 → 0.18.
  - Mechanism: the modes have different working sets (DS gathers x, 4 B/unit; SS does atomics on state plus a touched flag, 8 B/unit), so they leave the 3 MB L2 at different N. SS per-message cost jumps 5× from 2^18 to 2^19, while DS stays L2-resident until 2^20.
  - Above L2, DS cost keeps rising (0.23 → 0.64 ns/edge), likely from DRAM and TLB reach. There is no plateau.
- **Prediction 2 (CPU: s*(2^22) ≥ 1.15 × s*(2^20)): SUPPORTED.** +90%. DS per-edge cost rises 1.06 → 2.48 ns as the working set leaves the 12 MB L3.
- **Prediction 3 (s* within 35% of χ_op): SUPPORTED, 11/12.**
  - Finite cases give s*/χ_op = 0.78–1.06.
  - Miss: GPU at 2^17. χ_op = 0.73 predicts a crossover, but SS never loses. At this size fixed launch overheads dominate both modes, so linear per-element costs don't apply.
  - *Circularity note:* χ_op is computed from the same runs (DS median and SS cost at s ≥ 0.1), so this tests the linearity of SS time in m, not true out-of-sample prediction.
- **Conclusion:**
  1. The crossover is governed by the cost ratio at the operating point, χ_op(H, W). That ratio depends on **each mode's working set relative to each cache level**, so the regime boundary can be **non-monotonic in problem size** (GPU) or shift strongly (CPU, +90% across L3).
  2. A single "hardware constant" χ is insufficient.
  3. This is a candidate headline result for the paper, *if it survives an out-of-sample test*.
- **Next (EXP-006, to pre-register):** measure memory-hierarchy cost curves *independently of the workload*: random-gather throughput and random-atomic throughput vs working-set size, per platform. Predict χ_op(N), and hence s*(N), from those curves alone, then compare with EXP-004's measured s*. That is the real out-of-sample test of the hardware-relative claim.

## EXP-006 — out-of-sample prediction of s*(N) from workload-independent memory primitives (PRE-REGISTERED 2026-10-04, before running)

- **Primitives** (`harness/exp006_micro.py`): E = 2^25 random indices, uniform over a working set of M = N units, M = 2^17…2^22. No graph, no frontier, no step semantics.
  - GPU: `gather` y[i] = w[i]·x[idx[i]]; `atomic` = atomicAdd(float) + atomicExch(int) on a random index; empty-kernel launch overhead.
  - CPU: `gather`; `stream` (8-byte record copy); `rmw` (s[v] += w plus a flag check/set, v random within the thread's own block of M/6).
- **Composition (fixed now; no EXP-004 numbers are used):**
  - GPU:
    - c_e = gather(M=N), c_m = atomic(M=N)
    - T_DS = 5·launch + N·F·c_e; T_SS(m) = 3·launch + m·F·c_m
    - s* = (T_DS − 3·launch) / (N·F·c_m)
  - CPU:
    - c_e = gather(M=N), c_m = 2·stream + rmw(M=N)
    - (count pass ≈ 0.5 stream, routing ≈ 1 stream, apply read ≈ 0.5 stream, plus the random RMW)
    - s* = c_e / c_m
- **Target:** measured s* from EXP-004 (F = 32, random destinations, 6 sizes × 2 platforms = 12 cases). A predicted s* > 1 counts as a match where measured is ∞.
- **Success criteria:**
  1. Median |relative error| ≤ 0.30 over the 12 cases.
  2. Shape: the predicted GPU s*(N) has its minimum within one size step of the measured minimum (2^19).
  3. Predicted CPU s* rises by ≥ 30% from 2^20 to 2^22 (measured: +90%).
- **Interpretation rule:** passing means the regime boundary can be predicted from hardware primitives measured *without the workload*, the strongest form of the hardware-relative claim available on this machine. Failing means χ_op needs workload-specific terms, and we diagnose which primitive mis-predicts.

### EXP-005 — RESULT (synthetic graphs only)

- **Process notes (three measurement defects found and fixed before trusting any number):**
  1. The roadNet-CA download from SNAP stalled twice (the first time a 1 h hang with no timeout), so prediction 3 is **untested**.
  2. The first parser took Galois's `Sanity check, Time` line (1 ms). Discarded.
  3. Galois per-region `Time` stats are not comparable across modes: on syn_w256, Sync reported `Sync, Time` = 14 ms while the whole algorithm (`Timer_0`) took 761 ms. **The final run uses `Timer_0`** (lonestar's wrapper around the algorithm call). The parser now fails loudly unless exactly one such line exists.
- **Results** (7 paired trials, median; q25–q75 in brackets). All four graphs have the same N = 2^20, M = 8.4 M edges and F = 8. **Only locality differs, and through it dependency depth:**

| graph | BFS rounds | mean frontier | BFS T_sync / T_async (ms) | **BFS G_async** | SSSP G_async |
|---|---|---|---|---|---|
| syn_w16 | 74,953 | 0.001% | 9471 / 58 | **163** [159–174] | 1.02 |
| syn_w256 | 4,621 | 0.02% | 663 / 26 | **25.5** [22.7–27.0] | 1.01 |
| syn_w4096 | 289 | 0.35% | 55 / 23 | **2.62** [2.34–2.70] | 1.00 |
| syn_w0 (random) | 11 | 9.1% | 38 / 50 | **0.74** [0.71–0.83] | 1.01 |

- **Prediction 1 (BFS G_async monotone in rounds): SUPPORTED, strictly monotone.** It spans **0.74× → 163×**, more than two orders of magnitude, at identical size, edge count and total work.
  - Sync per-round cost in the barrier-dominated graphs: T_sync/R = 126 µs (w16), 143 µs (w256), 190 µs (w4096). That is Galois's per-round runtime overhead (worklist swap, parallel-loop launch), far above a raw 6-core barrier.
  - The async gain is essentially the avoided per-round overhead: R × ~125 µs, compared against ~25–60 ms of async work.
- **Prediction 2 (random graph: G_async ≤ 1.1): SUPPORTED.** BFS 0.74 (async *loses*, consistent with ω > 1: async does more work per vertex than level-synchronous BFS on a low-diameter graph); SSSP 1.01.
- **Prediction 3 (roadNet-CA): UNTESTED** (download failure).
- **Prediction 4 (SSSP same ordering): NOT SUPPORTED.** SSSP G_async = 1.00–1.02 on all graphs. Galois `deltaStepBarrier` vs `deltaStep` does not isolate barrier cost: delta-stepping already synchronizes per bucket, not per hop. With weights 1–100 the bucket count is far smaller than the hop depth, so barriers are rare in both variants. (This was flagged as a weaker claim in advance.) Not diagnosed further.
- **Conclusion:**
  - The asynchrony component is governed by **dependency depth and per-round work**, not by overall sparsity or total work. Identical workloads by size and sparsity span 0.74×–163×.
  - Asynchrony **loses** when rounds are few and large.
  - The *magnitude* is runtime- and hardware-relative: it scales with the per-round synchronization cost B. That is ~125 µs for the Galois runtime, but would be very different for a hardware barrier on a neuromorphic chip. That supports the same framing as EXP-004.
- **Caveats:**
  - Galois Sync and Async differ in more than barriers (pre-registered caveat), so this is "Galois async design vs Galois sync design". The ω > 1 explanation for the random-graph loss is inferred from the iteration counts (Async 1,058,840 vs Sync 1,048,240 on syn_w256), not measured on syn_w0.
  - Our own AE kernel is still needed for a clean decomposition with a controlled barrier cost.

### EXP-006 — RESULT (pre-registered criteria: 1 of 3 passed; GPU passes, CPU fails)

- **Process note:** the first launch failed before running, because WSL's outer shell split a grep pattern. Rerun via `harness/run_exp006.sh`.
- **Primitives** (`results/exp006_micro.json`; E = 2^25):
  - GPU gather: 0.084, 0.058, 0.068, 0.235, 0.518, 0.642 ns for N = 2^17…2^22.
  - GPU atomic: 0.142, 0.262, 1.757, 2.518, 3.051, 3.372 ns.
  - CPU gather: 0.93 → 3.24 ns. CPU rmw: 0.92 → 7.16 ns. CPU stream: ≈ 0.78 ns per record.
- **Predicted vs measured s\*** (measured from EXP-004):

| log2N | GPU pred | GPU meas | err | CPU pred | CPU meas | err |
|---|---|---|---|---|---|---|
| 17 | 0.805 | ∞ | miss | 0.375 | 0.121 | +211% |
| 18 | 0.279 | 0.270 | +3% | 0.366 | 0.122 | +199% |
| 19 | 0.043 | 0.040 | +9% | 0.406 | 0.126 | +223% |
| 20 | 0.095 | 0.079 | +20% | 0.380 | 0.127 | +199% |
| 21 | 0.171 | 0.126 | +35% | 0.371 | 0.172 | +116% |
| 22 | 0.191 | 0.177 | +8% | 0.372 | 0.241 | +54% |

- **Criterion 1** (median |error| ≤ 0.30): **FAIL** overall (0.54), driven entirely by CPU. GPU alone: median 0.09 over the 5 finite cases, all ≤ 0.35.
- **Criterion 2** (GPU argmin within ±1 of 2^19): **PASS.** The predicted minimum is exactly 2^19. **The non-monotonic GPU regime boundary is reproduced from workload-independent primitives.**
- **Criterion 3** (predicted CPU rise ≥ 30% from 2^20 to 2^22): **FAIL.** Predicted −2%; measured +90%.
- **Diagnosis:**
  - CPU **dense** is predicted well: gather 0.93 vs measured DS 0.90 ns/edge at 2^17.
  - The CPU **sparse** composition is wrong. Measured SS cost per message (6.2–10.9 ns) is 2.5× the composed c_m at small N.
  - The "stream" primitive is purely sequential, but the real SS reads each active unit's F contiguous out-edges at a *random* offset: short random segments, two cache lines per unit. It also has per-step bookkeeping (count pass, owner offsets, frontier compaction) that no primitive covers.
  - Because this missing cost is roughly independent of N, it also dilutes the N-dependence, which explains the failed criterion 3.
  - The GPU composition works because GPU SS cost is dominated by the atomic, which the primitive captures directly.
- **Conclusion:**
  - **Strong support on GPU:** the crossover, including its non-monotonic dependence on problem size, is predictable from hardware primitives measured without the workload (4/5 finite cases within 20%, all within 35%, correct argmin). The GPU 2^17 miss is the fixed-overhead regime, where linear per-element costs don't apply (also noted in EXP-004).
  - **Failure on CPU:** the primitive set is incomplete for a multi-pass software sparse kernel. Any CPU fix is **post hoc** and must be pre-registered and tested on fresh data (EXP-007), not tuned on these numbers.
- **Next (EXP-007, to pre-register):** add a "segmented random read" primitive (F contiguous elements at random offsets) and a per-step bookkeeping term. Fix the composition *before* measuring, then test on fresh seeds and F ∈ {16, 48}.
