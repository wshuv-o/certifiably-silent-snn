# Literature Review (Phase 2) — v1, 2026-10-04

Status key: **V** = verified against primary source · **P** = partially verified · **U** = from memory, verify before citing.

## 1. Closest work (novelty threats)

| Threat | Paper | What it already owns | What it lacks (our room) |
|---|---|---|---|
| HIGH | **NeuroScale** — Li, Imam, Manohar, *Nat. Commun.* 16:10329 (2025), doi:10.1038/s41467-025-65268-z **V** | Local handshakes replace global barriers; 4.27×/4.11× vs TrueNorth/Loihi-style sync at 16k cores; gain grows with sparsity (Fig 3d) and locality (Fig 3c); fades under dense global traffic; +17–25% energy | No sparsity vs asynchrony split; no burstiness, fan-out, dependency depth; not hardware-relative |
| HIGH | **Floorline** — Yik et al., arXiv:2511.21549 (2025) **V** | Aggregate sparsity/op counts fail to predict runtime; memory/compute/traffic-bound states on AKD1000, Speck, Loihi 2; runtime ∝ max synops/core | Single (time-stepped) semantics; no async axis, burstiness, dependency structure, cross-hardware ratios |
| HIGH | **Loihi 2 runtime model** — Timcheck, Pierro, Shrestha, arXiv:2601.10035 (2026) **V** | Analytical max-affine compute+NoC runtime bound, r ≥ 0.97 on hardware | One platform, sync semantics only |
| HIGH | **Ligra** (Shun & Blelloch, PPoPP 2013) / **Direction-optimizing BFS** (Beamer et al., SC 2012) **U** | Runtime sparse/dense switching by frontier-size threshold | Graph-only, sync-only, CPU; no asynchrony or neuromorphic cost model |
| HIGH | **PowerSwitch** — Xie et al., "Sync or Async: Time to Fuse…", PPoPP 2015 **U** | Adaptive sync/async switching in graph processing | Distributed CPU graphs; no sparsity/async decomposition or event-driven hardware |
| HIGH | **PolyGraph** (ISCA 2021) **U** | Graph accelerator design space incl. sync vs async, push/pull | Graph-specific; no general workload-space theory |
| MED | **Yan, Bai, Tang, Wong**, *IEEE TCAD* (arXiv:2409.08290) **V** | Analytical crossover: SNN (T=5) needs spike rate < 5.7% to beat equivalent QNN | Sparsity only, no asynchrony |
| MED | **Dampfhoffer et al.**, *IEEE TETCI* 7(3) 2023 **P** | SNN wins only at ~0.15–1.38 spikes/synapse/inference; memory access dominates | Sparsity only |
| MED | **Bhattacharjee et al.**, arXiv:2309.03388 (2023) **V** | Hardware overheads erase estimated SNN gains | Qualitative |
| MED | **Koopman et al.**, *TMLR* 2025 (arXiv:2408.05098) **V** | Async benefit depends on training: layer-sync-trained nets gain nothing async; async-aware training → up to 2× faster | Confound we must control |
| MED | **Chen et al.**, async multi-core SNN accelerator, arXiv:2407.20947 **V** | Barrier removal: 1.86× speed, 1.55× energy efficiency | No regime map |
| MED | **Sun et al.**, dual memory pathways, *Nat. Mach. Intell.* 8:901 (2026) **V** | Best design mixes sparse event + dense state paths | No general crossover |

## 2. Supporting / context (low threat)

- *Neuromorphic computing at scale* — Kudithipudi et al., *Nature* 637:801 (2025) **V** — frames "sparsity is key"; motivation.
- NeuroBench — Yik et al., *Nat. Commun.* 16:1545 (2025) **V** — hardware-independent proxies (activation sparsity, synops) are exactly what we argue are insufficient.
- Speck — Yao et al., *Nat. Commun.* 15:4464 (2024) **V** — fully async point; 0.42 mW resting.
- GALS SNN-training chip — Li M. et al., *Nat. Commun.* 17:4403 (2026) **V** — gains attributed to sparsity, GALS for scalability.
- Macroscopic brain simulation — Zheng et al., *Nat. Commun.* 16:9424 (2025) **V** — hybrid GPU+chip pipeline +2.1×; bottlenecks move to host side.
- HiAER-Spike — Frank et al., *npj Unconventional Computing* (2026) **V** (not Nat. Commun.) — hierarchical routing, 1 ms time-stepped.
- Tianjic (*Nature* 2019) **V**, Darwin3 (*Natl Sci Rev* 2024) **V**, DYNAP-SE2 (*NCE* 2024) **V**, SpiNNaker2 (arXiv 2401.04491) **V**, Hala Point (press release only) **P**, TrueNorth / Loihi 1 **U**.
- ANCoEF async simulator, arXiv:2411.06059 **V** — possible simulation tooling.
- Random walks on TrueNorth/Loihi — Smith et al., *Nat. Electron.* 2022 **U**; Davies et al., Loihi survey, *Proc. IEEE* 2021 **U** (reports where Loihi wins/loses); clock- vs event-driven SNN simulation — Brette et al. 2007 **U**; Sparseloop (MICRO 2022) **U**; parallel discrete-event simulation (conservative vs optimistic sync) **U**.

## 3. Synthesis

**Established**
1. Removing global barriers helps, more so with sparse activity, good locality and large systems — and can cost energy.
2. Aggregate sparsity does not predict deployed runtime; chips are compute/memory/traffic-bound and governed by per-core max load.
3. Sparsity-only break-even thresholds exist (≈ < 6% spike rate; ≈ 0.15–1.4 spikes/synapse).
4. Async benefit depends on the algorithm/training, not only hardware.
5. Graph processing already does adaptive sparse/dense and sync/async switching.
6. Leading designs are hybrid sparse/dense.

**Saturated** — "another neuromorphic accelerator"; "async beats barriers"; "sparsity alone isn't enough"; sparsity-only energy break-even; runtime switching per se.

**Unresolved (our opportunity)**
- No work separates the **sparsity gain** (dense-sync → sparse-sync) from the **asynchrony gain** (sparse-sync → async) in one framework.
- No **joint phase diagram** over sparsity × burstiness × locality × fan-out × dependency depth × compute intensity.
- No crossover expressed in **hardware cost ratios** that transfers across chips (existing models are single-platform or single-semantics).
- Burstiness and dependency structure are essentially unstudied as independent variables.
- No parameterized workload generator for this question.

**Required positioning**
- vs Floorline / Loihi 2 model: we add the execution-semantics axis (async) and make the model cross-hardware.
- vs NeuroScale: we generalize its sparsity/locality trends into a decomposed, hardware-relative crossover with more dimensions.
- vs Ligra / PowerSwitch / PolyGraph: switching is not our claim; the decomposition and the neuromorphic cost regime are.
- Control the training confound (Koopman et al.) — use workloads whose semantics don't depend on training, or train both ways.

## 4. Still to verify before citing

All **U** items (graph-side papers especially), HiAER-Spike volume/article no., SpiNNaker2 OJCAS DOI, Yan et al. TCAD DOI, "15–40% of A100" figure in the GALS training chip paper.

## Addendum (2026-10-06, overnight; 7 web searches, no agents) — related work for the N3 paper

- **NEST distributed simulation** exchanges spikes every minimum synaptic delay; neuron dynamics are decoupled within that interval (Hahne et al. 2020 Front. Neuroinform. 14:12 — [PMC7214808](https://pmc.ncbi.nlm.nih.gov/articles/PMC7214808/); the classic reference is Morrison et al. 2005, to be verified).
  → Our certificates give **lookahead from neuron dynamics**, which is useful exactly when delays are 1 step.
- **"The silence of the neurons"** (Heidarpur, Ahmadi, Ahmadi, Front. Neurosci. 2024 — [PMC10933060](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933060/)): skips *computation* of Izhikevich nonlinear terms in quiet periods. It is **approximate** (no guarantee; small spike-timing deviations), with STDP retraining, 20–90% compute savings and FPGA speed-ups.
  → Distinct: ours is **exact**, certifies *silence* (not quiescence of state), and targets *synchronization*.
- **Koopman et al., TMLR 2025** — "Exploring the limitations of layer synchronization in SNNs" ([OpenReview](https://openreview.net/forum?id=mfmAVwtMIk)): "unlayered backprop" trains models for asynchronous processing, which **changes the network semantics**.
  → Distinct: ours keeps lock-step semantics **bit-exactly** and removes waiting through certificates.
- **Formal verification of SNNs:** timed automata (De Maria et al.), probabilistic model checking / DTMCs ([arXiv:2506.13340](https://arxiv.org/html/2506.13340), [arXiv:2606.20674](https://arxiv.org/pdf/2606.20674)), and verification of spiking controllers for CPS ([arXiv:2408.01996](https://arxiv.org/pdf/2408.01996)).
  → Distinct: these verify properties of fixed networks (with state-space explosion); we **train** networks so a cheap per-neuron bound certifies silence at run time.
- **Loihi barrier mechanism:** barrier messages exchanged between neighbouring cores flush in-flight spikes and propagate the time-step advance (Davies et al. 2018). Loihi 2: sub-200 ns chip-wide time steps, 3D multi-chip meshes ([Intel brief](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf)).
- **Conservative vs optimistic PDES for SNNs:** conservative (YAWNS-style in ROSS) is usually faster ([ACM TOMACS 3649464](https://dl.acm.org/doi/10.1145/3649464)).
- **Verdict:** no prior work found that *trains* SNNs so that their silence is *provable*, or that uses dynamics-derived certificates for exact synchronization-free execution. Novelty holds at 18 searches in total; the claim stays "to our knowledge".

## Addendum 2 (2026-10-06): positioning facts for the paper (4 searches)

- **SHD state of the art ≈ 95–96%:**
  - Sun et al. 2025: 96.26% (delays + attention);
  - Schöne et al. 2024: 95.9% (event-based SSM);
  - Baronig et al. 2024/2025: 95.8% (RSNN, adaptive LIF, symplectic Euler; Nat. Commun. 2025, "Advancing spatio-temporal processing through adaptation" — [link](https://www.nature.com/articles/s41467-025-60878-z));
  - Hammouamri et al. 2023: 95.1% (learned delays);
  - leaderboard: [zenkelab.org](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
  - **Our 82% single-layer ALIF is far below SOTA.** We state this, and add S2b (a stronger recipe) to narrow the gap.
- **Interconnect latency anchors for the emulated-latency sweep:**
  - MPI small-message latency: ~1–7 µs on InfiniBand (sub-µs on NDR), ~30 µs on commodity Ethernet, 2–5 µs on tuned RoCEv2 (RDMA-MPI literature; e.g. Liu et al., "High performance RDMA-based MPI implementation over InfiniBand", ICS 2003 — [ACM](https://dl.acm.org/doi/10.1145/782814.782855)).
  - Loihi 2: chip-wide time steps < 200 ns; asynchronous chip-to-chip links; Kapoho Point (8 chips), Hala Point (1152 chips) ([Intel brief](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf)).
  - SpiNNaker inter-chip links: 250 Mbps self-timed ([Dugan 2013](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-cdt.2012.0139)).
- **Implication for claims:** the measured speed-up (1.35× at 5 µs, 1.50× at 20 µs, 1.55× at ≥ 100 µs) is relevant to **distributed/cluster SNN simulation and multi-board systems**, not to single-chip execution, where synchronization is sub-µs. The paper must say this explicitly.
