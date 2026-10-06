# Phase 4 — Workload Space & Cost Model (on paper) — v1, 2026-10-04

Purpose: define the "dataset" (a parameterized workload space), check which dimensions can be manipulated independently, and derive falsifiable predictions *before* any code.

---

## 1. Abstract workload model

A workload W runs on N stateful units (neurons / vertices / cells) connected by a directed graph with mean fan-out F. Execution proceeds in logical steps t = 1…T (or until a convergence criterion). At step t a set A_t of units is **active** (its state changed and must notify successors). Each active unit performs I operations and emits F messages.

This covers SNN inference, graph propagation (BFS/PageRank), relaxation solvers, cellular/dynamical systems with partial updates, and event streams. Dense GEMM is the special case A_t = all units, every step.

## 2. Workload dimensions

| Symbol | Dimension | Operational definition (measurable) | How we control it |
|---|---|---|---|
| s | Activity sparsity | mean \|A_t\| / N | input rate / threshold, or prescribed (open-loop) |
| τ | Temporal burstiness | CV of \|A_t\| over t | Markov-modulated activity process |
| κ | Spatial imbalance | mean over t of (max-core load / mean-core load) | clustering of active set × mapping |
| L | Locality | fraction of messages that stay on-core; mean hop count h | graph rewiring prob. (Watts–Strogatz-style) + mapping |
| F | Fan-out | mean out-degree | graph generator |
| D | Dependency depth | longest causal event chain per unit of work (critical path / total work) | layered depth vs recurrence; propagation rule |
| I | Compute intensity | ops per event | neuron/update model complexity (synthetic kernel) |
| M | State size | bytes of state per unit | synthetic padding / model |

## 3. Independence analysis

Not all dimensions are free. Known couplings:

1. **s ↔ F ↔ D (dynamics coupling).** In closed-loop propagation, s_{t+1} depends on s_t, F and thresholds; D emerges from graph structure. → **Two generator modes:**
   - **Open-loop (controlled):** active sets are sampled from a prescribed stochastic process independent of the graph. Lets us vary s, τ, κ, L, F, I, M factorially. D is set by imposing explicit dependency chains.
   - **Closed-loop (realistic):** activity emerges from the update rule. Used to check that open-loop conclusions survive real dynamics.
2. **L ↔ κ (mapping coupling).** Placing communicating units together raises L but can concentrate activity on few cores (raises κ). → Vary graph rewiring (changes L) and mapping policy (changes κ) as separate factors; report the achieved (L, κ) pairs rather than nominal settings.
3. **τ ↔ κ.** Temporal bursts are often spatially clustered. → Generate temporal and spatial structure with separate processes in open-loop mode.
4. **I ↔ M.** Richer update models usually carry more state. → Synthetic kernel decouples them.

Rule: every experiment reports the **measured** values of all eight dimensions, plus their correlation matrix across the sweep, so confounding is visible rather than assumed away.

## 4. Three execution modes

| Mode | Semantics | Pays for |
|---|---|---|
| **DS** dense-sync | update every unit every step; barrier per step | all N units and N·F edges, barrier |
| **SS** sparse-sync | update only A_t (active set / frontier); barrier per step | active work at sparse-access cost, frontier construction, load imbalance, barrier |
| **AE** async event-driven | units fire when inputs arrive; no global barrier | active work, per-event queue/scheduling, possible extra work, dependency stalls |

Gain decomposition (the core of the paper):

  **G_total = T_DS / T_AE = (T_DS / T_SS) × (T_SS / T_AE) = G_sparsity × G_async**

## 5. Cost model

Hardware H is described by per-operation costs (time; same structure for energy):

| Symbol | Meaning |
|---|---|
| c_d | dense per-unit update (streaming, vectorized) |
| c_e | dense per-edge cost (e.g., GEMV element) |
| c_s | sparse per-unit update (indirect access) |
| c_f | frontier/active-set bookkeeping per active unit |
| c_m(h) = c_m0 + h·c_hop | per-message routing cost, depends on locality |
| c_q | async queue/scheduling cost per event |
| B(P) | barrier cost with P cores |
| ℓ | latency of one dependent event hop |

Per step, with P cores, W_t = \|A_t\|·(c_s + c_f + F·c_m) the active work:

- T_DS ≈ N(c_d + F·c_e)/P + B
- T_SS ≈ κ·W_t/P + B
- T_AE ≈ max( ω(1+q)·W_t/P , critical-path term D·ℓ ), with q = c_q/(c_s + F·c_m) and ω ≥ 1 the extra work async may do before reaching the same solution

### Dimensionless groups

| Group | Definition | Meaning |
|---|---|---|
| χ | c_e / c_m | dense-edge vs event cost — **the key hardware ratio** |
| β | B·P / W_t | barrier cost vs useful work per step |
| κ | as above | load imbalance |
| ω, q | as above | async overheads |
| δ | D·ℓ·P / W_total | how much of the run is dependency-bound |

## 6. Predictions (falsifiable)

**P1 — Sparsity crossover is hardware-relative.** For F ≫ 1, G_sparsity = 1 at
  s* ≈ (c_d + F·c_e) / (κ·(c_s + c_f + F·c_m)) ≈ χ / κ.
Sparse execution should pay off below s*. On GPUs (cheap streaming, expensive scatter) χ is small, so s* should be small. On event-native hardware χ ≈ 1, so s* ≈ 1/κ. *Test:* measured s* across ≥ 2 platforms should scale with measured χ.

**P2 — Asynchrony gain grows as activity gets sparser.** When δ is small:
  G_async ≈ (κ + β) / (ω(1+q)).
β ∝ 1/s, because barrier cost stays fixed while work per step shrinks. So the two components **interact**: sparsity makes barriers relatively costlier, which makes asynchrony worth more. This is consistent with NeuroScale Fig 3d; we predict the functional form.

**P3 — Asynchrony loses when dependencies dominate or overheads exceed imbalance.** G_async < 1 when ω(1+q) > κ + β, or when δ → 1 (deep serial chains: no slack to run ahead).

**P4 — Sparsity alone is insufficient.** Workloads with equal s but different (κ, L, D, τ) should fall in different regimes. *Falsification test:* if a model using s alone predicts mode choice about as well as the full model (compare R² or classification accuracy on held-out workloads), the core hypothesis is rejected.

**P5 — Negative control.** Dense GEMM (s = 1, κ = 1, β ≈ 0): G_sparsity < 1 and G_async ≤ 1 on every platform.

**P6 — Time vs energy diverge.** Because async trades barrier time for queue and routing energy, the energy crossover should sit at lower s than the time crossover. NeuroScale's +17–25% energy is one data point for this.

## 7. Avoiding simulator circularity

The model must be tested on **real execution**, not only in a simulator parameterized with our own constants:

1. **Real CPU (multicore):** implement DS / SS / AE (AE via lock-free queues / work-stealing), and measure c_* with microbenchmarks.
2. **Real GPU:** DS (dense GEMV) / SS (frontier kernels) / AE (persistent-thread queue), and measure c_*.
3. **Neuromorphic:** parameter sets from published characterizations (Loihi 2 runtime model, Floorline, NeuroScale), used for prediction only. Real Loihi 2 access (Intel INRC) would be ideal, but its availability is unknown.

Measure the c_* constants once per platform → predict s*, G_async → compare with measured runs. The model is only credible if the predictions transfer across platforms.

## 8. Metrics (at matched solution quality)

Time-to-solution, operations-to-solution, memory traffic, messages, barrier time, active-compute fraction, and energy where measurable. Stopping rules:
- **Iterative workloads:** stop at the same residual tolerance.
- **SNN:** identical output for deterministic semantics; otherwise matched accuracy, with the training confound controlled (Koopman et al.).

## 9. Benchmark workloads → dimensions they exercise

| Workload | Main dimensions | Role |
|---|---|---|
| Synthetic open-loop | all, factorial | build the phase diagram |
| Synthetic closed-loop | s, F, D emergent | realism check |
| BFS / SSSP | s varies over time (frontier), D | graph regime; compare against Ligra-style switching |
| PageRank / Jacobi relaxation | ω (async convergence), s decays | async work inflation |
| Random walks / Monte Carlo | low s, low D, high τ | expected async-friendly |
| SNN inference (event data) | τ, s; training confound | neuromorphic home turf |
| Partial-update dynamical system (e.g., cellular automaton / reaction-diffusion with local activity) | L, κ | locality / imbalance |
| Dense GEMM | s = 1 | negative control |

## 10. Platform decisions (resolved 2026-10-04)

**Available machine:** laptop with an Intel i7-10750H (6 cores / 12 threads), NVIDIA RTX 2060 Laptop (6 GB, driver 565.90), 23.8 GB RAM, Windows 11, Python 3.14. No CUDA toolkit or C++ compiler is installed.

- **Two real platforms with very different χ (CPU vs GPU).** That is enough to test P1's claim that the crossover is hardware-relative. Neuromorphic chips enter through published parameter sets, used for prediction only. *Optional:* apply for Intel INRC (Loihi 2 cloud access) as a later validation.
- **Implementation:** timing must not be dominated by interpreter overhead, so kernels need to be compiled.
  - CPU: Numba (`parallel`/`prange` for DS and SS, threads plus lock-free queues for AE), or C++ if MSVC Build Tools are installed.
  - GPU: CuPy `RawKernel` (CUDA C through NVRTC, so no full toolkit needed) for DS/SS/AE, with AE as a persistent-thread work queue.
  - Check that Python 3.14 wheels exist for Numba and CuPy; fall back to a Python 3.12 venv if not.
- **Energy:** GPU energy via NVML (`nvidia-smi` / pynvml power sampling). CPU package energy on Windows is not easily readable (RAPL), so it is optional, via LibreHardwareMonitor. **Time-to-solution is the primary metric; energy is secondary and GPU-only unless CPU readout works.**
- **Laptop caveats:**
  - Thermal and boost-clock variance: we need repeated runs (≥ 5), a warm-up, and reported confidence intervals, and we log clocks and temperature during runs.
  - AC power and the maximum-performance power plan must be fixed.
- **Limitation (to state in the paper):** the CPU has only P = 6 cores, so barrier effects at neuromorphic scale (thousands of cores) can't be measured on the CPU. The GPU covers large P, and the model extrapolates beyond that, checked against NeuroScale's scaling numbers.
- **Headline figure:** s × β phase diagram coloured by the best mode (DS/SS/AE), one panel per platform, with the predicted boundaries overlaid.

## Next step

Answer §10, then **Phase 6 (reference implementations):** find existing code for frontier/async graph kernels (Ligra, Gunrock, Galois) and event-driven SNN simulators to reuse for SS/AE baselines, so we don't build weak strawman implementations.
