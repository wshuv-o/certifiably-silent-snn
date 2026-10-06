# Project Brief (Layer 2)

Last updated: 2026-10-04

## STATUS 2026-10-06 07:15 — manuscript complete (`research/paper/manuscript.docx`)

**Title v4 (final for submission):** *Certifiably Silent Spiking Networks: Bounding Excitation Makes Recurrent SNN Silence Provable and Reduces Synchronization in Exact Multi-Core Execution*

- **Core contribution:** the excitatory-drive budget principle (R_i vs (1 − β)·θ) + silence certificates with proofs + exact certificate-based execution. The certificate loss is one of three enforcement methods (with L1 and a cap).
- **Main limitations:** cost grows with size and on SSC; no hardware; sub-SOTA accuracy; dense speed-up is small.
- See `research/paper/README_MORNING.md` for the full summary, risks and next steps.

## CURRENT DIRECTION (2026-10-05): N3 — Certifiably Silent Spiking Networks

**Working title v3 (provisional):** *Certifiably Silent Spiking Networks: Training Recurrent SNNs for Provably Synchronization-Free Execution*

- **Mechanism:** a training loss that bounds each neuron's worst-case K-step membrane reach under recurrent input, so that quiet periods become *provably* quiet and synchronization can be skipped **exactly**.
- **Evidence so far (PILOT-004, SHD, local connectivity, 1 seed):**
  - provable 4-step neighbour silence rises from 0% to 56% (oracle 61%), at −2.2 points accuracy;
  - 0 soundness violations;
  - the network still relies on recurrence (−8.2 points when it is removed).
- **Prior art to cite and distinguish:** S-IBP / S-CROWN (certified robustness for SNNs, same bounding family, different objective); Time Warp / PDES for SNNs; Spike-simulator timestep grouping; NeuroScale (hardware local synchronization).
- **Next:**
  1. ~~Seed replication~~: **done, replicated on 3/3 seeds** (57–58% certified at K = 4, mean accuracy change −0.2 points, 0 violations).
  - **S1c (compiled 8-core engine, Lemma-1 certificates): PASS.** 1.50–1.55× exact wall-clock speedup over local handshake at L ≥ 20 µs. Untrained nets are certifiable only 0.2% of the time vs 67% after training, so the whole gain is due to training.
  - **S2 (512 ALIF, dense, GPU): PARTIAL (moderate regime, control 82.0% < 85%).** λ = 0.3: −0.66 points, 0% → 59% core-certified (95% of oracle), 0 violations, R ≈ threshold. Seed replication (S2-rep) running.
  - **Scale-up phase declared** (`N3_SCALEUP_PLAN.md`): soundness lemmas written, plus the "certifiability threshold" proposition (R_i < (1 − β)·θ − ā ⇒ unbounded horizon).
  - **S1 wall-clock engine (exact, 8 processes): INCONCLUSIVE.** Training gain 1.26× under the cert protocol, but cert is *slower* than a local handshake (Python certificate overhead dominates at ≤ 20 µs latency). Next: S1b with a C++/compiled engine, an O(1) threshold certificate and a latency sweep (to be pre-registered).
  - **PILOT-007 (2026-10-05):** the sparse-regime test reached training gains of only 1.44× / 1.57× (< 2.0 bar), so **STOP**. N3 is to be written up as a modest short paper. The rate-target manipulation failed to lower firing rates (stated limitation, no rerun). Final claims list: `BRAINSTORM_003_NEUROMORPHIC.md` → "N3 — FINAL STATUS".
  - **PILOT-006 (synchronization messages): INCONCLUSIVE.** Certified 2.17× fewer messages vs lock-step, but untrained networks already get 1.52× (free 1-step certificates). The gain from training is ~1.43×, and the oracle ceiling is only ~3× in this regime. **Benefit real but modest.**
  2. Stronger models and a second dataset (disk-limited).
  3. An execution engine that actually skips synchronization, benchmarked against per-step and fused baselines.
  4. Runtime cost of the certificate.
  5. Test the E/I-balance hypothesis.

## (History) STATUS: MEASUREMENT TRAP (flagged 2026-10-04)

**Experiments are paused until a method contribution is defined.**

- The project drifted into the measurement trap (Research OS guardrail). EXP-001…006 measure *existing* execution strategies (dense sweep, frontier push, Galois sync/async) with existing tools, and title v2 below presents those measurements as the contribution.
- Brainstorm test: all three headline "findings" fail it.
  1. Cache-dependent crossovers are expected from roofline-type reasoning, Sparseloop and SpMV literature.
  2. Async BFS gains growing with diameter are known from the Galois and PowerSwitch line of work.
  3. "Sparsity is not enough" is already claimed by Floorline (Yik et al. 2025).
- **The experiments stay valuable as motivation/analysis and as a validated harness, but they are not the paper.**
- Required before any further experiments: return to Phase 1 and Phase 5, and define a **new method, mechanism or theory** that these observations motivate. It must be evaluated as such. The human decides the direction.

**Phase 5 analysis (2026-10-04): `PHASE5_MECHANISM_ANALYSIS.md`.**
- Regret analysis shows that "smart switching" has single-digit-percent headroom, so it is ruled out.
- SNN timestep grouping and temporal aggregation are taken by prior art.
- Six expectations are set that any mechanism must meet.
- One surviving candidate, speculative multi-step SNN execution on GPU, has medium–high novelty risk.
- User constraint: software-only (commodity GPU/CPU), start small.

**Candidate 1 (event-driven incremental encoding for streaming Whisper): DROPPED on 2026-10-04.**
- Pre-registered PILOT-001 found that new audio changes essentially every old encoder frame by 3–5% (fewer than 2.4% of frames stay under 1%), so there is nothing to skip.
- See `CANDIDATE_STREAMING_ENCODER.md`.
- Lesson: check the "units really stay unchanged" property with a cheap pilot before designing any event-driven mechanism.

**User requirement (2026-10-04): the work and headline must be deeply neuromorphic** (spiking or event-based computation at the core), software-only.
- Idea A (edit-aware KV reuse) dropped as not neuromorphic.
- Current candidates: `BRAINSTORM_003_NEUROMORPHIC.md`. N1 (hardware-calibrated spike budgets) and N2 (barrier-free exact event-driven SNN on GPU) lead.
- Literature check done (8 searches). N1 dropped (Floorline collision).
- **N2* tested in PILOT-003 (2026-10-05): not viable.** Provable partition-level silence ≈ 0% even though the layer is actually silent in 52% of 4-step windows. Worst-case bounds are far too loose; the optimistic alternative is Time Warp (prior art, ~2× headroom).
- PILOT-003b (local connectivity): bound nearly tight at K = 2 (62% vs 64% oracle) but 0% at K = 4, so **N2* closed** (pre-registered rule).
- Candidates exhausted so far: measurement study, streaming Whisper, edit-KV (not neuromorphic), N1, N2*. Disk on D: is now ~2 GB free.

## Working Title (v2 — REJECTED as measurement trap; kept for history)

**Sparsity Is Not Enough: The Memory Hierarchy and Dependency Depth Decide When Brain-Inspired Execution Wins**

Why revised (per Research OS Phases 1 and 3: the title is a working hypothesis and must follow the evidence):
- v1 said "Neuromorphic Parallel Execution", but all measurements are on CPU/GPU *emulating* brain-inspired execution principles. Neuromorphic chips enter only through published parameters. v1 overclaimed.
- The evidence now names the governing factors:
  - Sparsity gain: crossover set by mode-specific costs at the operating point of the memory hierarchy, non-monotonic in problem size on GPU (EXP-004, EXP-006).
  - Asynchrony gain: set by dependency depth vs per-round synchronization cost, 0.74–163× at equal size and work (EXP-005).
- Still provisional: CPU prediction failed (EXP-006), no own AE kernel yet, no neuromorphic hardware. If EXP-007 or the AE work contradict this, revise again.

Title history:
- v1 (2026-10-04, pre-data): "When Does Brain-Inspired Computing Win? Disentangling Sparsity and Asynchrony in Neuromorphic Parallel Execution"

Note: if claims extend to non-neural workloads (graphs, Monte Carlo), keep "neuromorphic" in the title and state in the abstract that brain-inspired execution principles are tested beyond neural networks.

## Field

Neuromorphic / brain-inspired computing; computer architecture; parallel execution models.

## Current Stage

Phase 2–4: literature mapping done (first pass), moving to the "on-paper" workload-space analysis. **No code yet.**

How the constitution maps here: the "dataset" is a **parameterized workload space**, not a fixed dataset. Phase 4 (data-first analysis) = formally defining workload dimensions, checking which can be manipulated independently, and testing on paper whether proposed measurements can distinguish the competing execution regimes.

## Research Question (v2)

When brain-inspired execution principles (sparse event-driven updates, barrier-free asynchrony) are run on real hardware, what sets the boundary where each one beats synchronous dense execution?

The evidence so far points to two governing factors:
- **Sparsity:** the ratio of dense-gather to event-update cost *at the current position in the memory hierarchy* (working set vs cache), which can make the boundary non-monotonic in problem size.
- **Asynchrony:** dependency depth relative to per-round synchronization cost.

Can both be predicted from workload-independent hardware primitives?

### Research Question (v1, superseded)

Which measurable properties of a workload, *relative to hardware cost ratios*, determine when brain-inspired sparse, asynchronous, event-driven execution beats synchronous parallel execution — and when it loses?

## Core Hypothesis (v2, falsifiable)

**H1 (sparsity component):** the dense-vs-sparse crossover is s* ≈ χ_op = c_dense-edge(M) / c_event(M). Both costs are evaluated at each mode's working set M relative to the cache hierarchy, so s* is predictable from workload-independent primitives.
- Supported on GPU (EXP-006).
- Failed on CPU (EXP-006). Must pass the pre-registered EXP-007 or be narrowed.

**H2 (asynchrony component):** G_async grows with dependency depth R and per-round sync cost B, roughly (R·B + W) / (ω·W), and drops below 1 when rounds are few and wide.
- Supported qualitatively on Galois BFS (EXP-005).
- Needs our own AE kernel with controlled B.

**H3 (sparsity is not enough):** workloads with equal sparsity land in different regimes depending on M (cache), L (locality) and D (depth).
- Supported (EXP-003, -004, -005). Still needs a formal sparsity-only-model comparison (Phase 4 P4).

### Core Hypothesis (v1, superseded)

The advantage of event-driven execution is governed not by sparsity alone but by the interaction of activity sparsity, temporal structure, locality, dependency structure, communication fan-out and computational intensity. The benefit decomposes into a **sparsity component** and an **asynchrony component**, each with a predictable crossover that depends on workload structure relative to hardware costs (Φ(W, H), not Φ(W)).

## Key Design Decisions

- **Three-way comparison**, not two: dense synchronous → sparse synchronous (active set + barriers) → asynchronous event-driven. Isolates sparsity gain vs asynchrony gain.
- **Hardware-relative model**: crossover expressed in terms of hardware cost ratios (event-routing cost vs dense op cost, etc.); cost constants grounded in measured/published hardware to avoid simulator circularity.
- **Equal-solution-quality metrics**: async execution changes convergence paths; compare time/ops/traffic-to-solution at matched quality, not per iteration.
- **Dense GEMM as negative control**: we expect to lose there; it establishes the boundary.

## Workload Dimensions (draft)

S activity sparsity · T temporal sparsity/burstiness · L memory locality · F fan-out · C communication intensity · D dependency structure/depth · I computational intensity per event · state size. Open question: which are independently manipulable (S, F, L, T are correlated in real workloads).

## Candidate Workloads

Synthetic parameterized generator first; then graph propagation (BFS/PageRank), sparse iterative linear algebra, Monte Carlo/random walks, dynamic systems with partial updates, event streams, SNN / sparse ANN inference, dense GEMM (negative control).

## Primary Metrics

Time-to-solution, operations-to-solution, memory traffic, communication volume, synchronization cost, active compute fraction, throughput; energy-to-solution once real hardware is available. Include conversion/preprocessing overheads (NeuroBench principle).

## Closest Work / Novelty Threats

**Superseded by `LITERATURE_REVIEW.md` v1 (verified).** Top threats now: NeuroScale (Nat. Commun. 2025), Floorline (Yik et al. arXiv 2511.21549), Loihi 2 runtime model (Timcheck et al. 2026), Ligra/Beamer BFS, PowerSwitch, PolyGraph. HiAER-Spike is npj Unconventional Computing, not Nat. Commun. Original list kept below for history.

Neuromorphic:
- NeuroScale (Nat. Commun. 2025) — decentralized async synchronization. *[verify]*
- NeuroBench (Nat. Commun. 2025) — benchmarking framework; dense vs effective ops.
- GALS multicore SNN-training architecture (Nat. Commun. 2026). *[verify]*
- Speck (Nat. Commun. 2024) — fully async sensing + compute.
- HiAER-Spike — hierarchical event routing at large scale. *[verify numbers/venue]*
- Brain-inspired macroscopic brain simulation, CPU/GPU/BI hybrid (Nat. Commun. 2025). *[verify]*
- Random walks on TrueNorth/Loihi (Smith et al., Nature Electronics 2022).
- Dual-memory algorithm/hardware co-design (Nat. Mach. Intell. 2026). *[verify]*
- SNN vs ANN energy break-even analyses (e.g., Dampfhoffer et al. 2022) — 1-D version of our phase diagram.

Outside neuromorphic (high threat to the crossover/adaptive-runtime claims):
- Direction-optimizing BFS (Beamer et al., SC 2012) and Ligra (Shun & Blelloch, PPoPP 2013) — runtime sparse/dense switching by frontier-size threshold.
- Sparseloop (MICRO 2022) / Timeloop — analytical modeling of when sparsity pays.
- GraphLab/PowerGraph async vs BSP comparisons.
- UpDown — fine-grained event-driven manycore for irregular workloads. *[verify venue/year]*

## Candidate Contributions

C1 workload formalization · C2 analytical (hardware-relative) crossover model · C3 parameterized workload generator/benchmark · C4 empirical phase diagram separating sparsity vs asynchrony gains · C5 predictor W→{sync, sparse-sync, event, hybrid} · C6 adaptive heterogeneous runtime (later; must differentiate from Ligra-style switching) · C7 real workloads.

## Known Weaknesses / Reviewer Attack List

- (A) Phase diagram may just reflect simulator cost constants → ground in hardware, parameterize by cost ratios.
- (A) "Crossover/adaptive switching already exists" (Ligra, Beamer) → contribution must be the sparsity/asynchrony decomposition and the neuromorphic regime, not switching per se.
- (B) Workload dimensions confounded → establish independent manipulability before experiments.
- (B) Async changes convergence → matched-quality metrics.
- (C) "Brain-inspired" framing vs non-neural workloads.
- (A) "Sparsity doesn't predict performance" and compute/memory/traffic-bound taxonomy already claimed (Floorline) → our novelty is the async axis + sparsity/asynchrony decomposition + cross-hardware ratios.
- (B) Async benefit depends on training (Koopman et al. 2025) → control training or use training-free workloads.
- (B) Async can cost energy (NeuroScale +17–25%) → report time and energy separately; crossover may differ per metric.

## Next Step

Phase 4 on-paper analysis done: `PHASE4_WORKLOAD_SPACE.md` (dimensions, independence analysis, three-mode cost model, predictions P1–P6). Key prediction: sparsity crossover s* ≈ χ/κ (χ = dense-edge/event cost ratio); asynchrony gain ≈ (κ+β)/(ω(1+q)) and grows as s falls. Hardware resolved: i7-10750H (6C) + RTX 2060 Laptop 6 GB; time primary, GPU energy via NVML; neuromorphic via published params (optional INRC Loihi 2 later). Phase 6 done: `PHASE6_REFERENCE_IMPLEMENTATIONS.md` — gapbs (CPU SS bar), Ligra (switching baseline), Galois (CPU AE), Gunrock + Atos (GPU SS vs AE), Spice/GeNN/Brian2 (SNN); own harness in Numba + CuPy; strawman guard ≤1.5× of references. Blocked on: installing WSL2 Ubuntu (references are Linux-only).

## Data Findings (from experiments)

- EXP-001 v2 (2026-10-04): DS vs SS crossover s* ≈ 0.08 (GPU, random dst), 0.09–0.19 (CPU), 0.18–0.57 (GPU, local dst). For F ≥ 32, s* ≈ χ = c_e/c_m within ~5% in 3 of 4 cases. Locality shifts s* in **opposite directions** on CPU vs GPU, an early sign that regime boundaries are hardware-relative. Not yet an out-of-sample test. See `EXPERIMENT_LEDGER.md`.
- EXP-002 (pre-registered, **failed**): constants fitted at one N do not transfer. χ is not a portable hardware constant.
- EXP-003: spatial concentration of activity shifts s* through locality (CPU sparse benefits, +12%) and atomic contention (GPU sparse suffers, −27%), **not** through load imbalance as modelled. "κ" as a single dimension conflates three mechanisms.
- EXP-004: s* tracks the operating-point cost ratio χ_op(H, W) (11/12 within 22%). χ_op depends on each mode's working set relative to each cache level, so the GPU regime boundary is **non-monotonic in problem size** (s* = ∞, 0.27, 0.04, 0.08, 0.13, 0.18 for N = 2^17…2^22). On CPU, s* rises 90% as the working set leaves L3. **Candidate headline result**, pending the out-of-sample EXP-006 (predict χ_op from workload-independent memory microbenchmarks).
- EXP-005 (Galois, independent code): the BFS asynchrony gain is strictly monotone in dependency depth, **0.74× → 163×**, at identical N, M, F and total work. Async *loses* on shallow, wide-frontier graphs. The magnitude scales with the runtime's per-round sync cost (~125 µs in Galois). SSSP pair is uninformative (delta-stepping already removes most barriers). roadNet-CA untested (download failed).
- EXP-006 (pre-registered out-of-sample): **GPU: crossovers predicted from workload-independent memory primitives**: 4/5 finite cases within 20%, all within 35%, and the non-monotonic minimum at 2^19 reproduced exactly. **CPU: failed** (sparse cost under-predicted ~2.5×; missing segmented-random-read and bookkeeping terms). The CPU fix must be pre-registered as EXP-007.
- **Revised core claim (provisional):** regime boundaries are set by mode-specific costs at the operating point of the memory hierarchy, not by sparsity or by fixed hardware constants.
- Laptop measurement protocol: CPU 6 threads + active OpenMP wait; GPU paired/interleaved DS–SS with warm-up (clock swings 300–1860 MHz otherwise).

## Next Steps (as of 2026-10-04, end of session 1)

1. **EXP-007:** pre-register a corrected CPU sparse composition (segmented random reads plus bookkeeping), test on fresh seeds and F ∈ {16, 48}.
2. **Own AE kernel** with a *controlled* barrier cost (CPU and GPU), so the sparsity × asynchrony decomposition is measured in one codebase, not via Galois's different designs.
3. Retry roadNet-CA (EXP-005 prediction 3) from a mirror, or with a longer timeout.
4. GPU strawman guard against Gunrock/Atos (still pending).
5. Then start drafting the figure plan (Phase 17). Candidate headline figures: GPU s*(N) non-monotonic (measured vs primitive-predicted); G_async vs dependency depth (0.74–163×).

## Manuscript Status

Not started. Target venue deliberately not chosen yet.
