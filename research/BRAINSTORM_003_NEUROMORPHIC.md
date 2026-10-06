# Brainstorm 003 — Deeply Neuromorphic, Software-Only (2026-10-04)

- **User requirement:** the work and the headline must be *deeply* neuromorphic. Spiking or event-based computation is the object of study, not a metaphor.
- **Constraints:** software only (RTX 2060 / 6-core CPU), small disk (~5 GB), start small.
- **Idea A** (edit-aware KV reuse) was **dropped**: its "reconsolidation" framing was decorative. PILOT-002 was stopped before producing results.
- **Method:** from knowledge only; no web. Novelty is **unverified** until a literature check, which needs user permission.
- **Our own data enters only as motivation** (measurement-trap rule): EXP-001…006 showed that the point where event-driven beats dense execution depends on hardware and on each layer's working set, and that barrier-free execution wins by orders of magnitude when per-step work is tiny and the dependency chain is long.

---

## N1. Hardware-calibrated spike budgets: put sparsity where it actually pays

- **Problem:**
  - SNNs are trained with a *uniform* spike-rate penalty and then reported in hardware-independent synaptic operations (NeuroBench).
  - But real speed and energy improve only for layers whose activity falls **below that layer's event-vs-dense crossover**. That crossover differs per layer (working set vs cache) and per platform (EXP-004/006; Floorline makes the same point for neuromorphic chips).
  - So uniform sparsification spends accuracy on layers that never cross, and under-sparsifies layers sitting just above their crossover.
- **Neuromorphic core:** spike-rate regulation as a training objective (homeostasis), with per-layer targets derived from the deployment hardware's measured event and dense costs.
- **Mechanism:**
  1. Calibrate per-layer crossover rates s*_l from workload-independent primitives. EXP-006 showed this is predictive on GPU.
  2. Train with a differentiable *deployment-cost* regularizer that charges each layer min(dense cost, event cost(rate)) rather than raw spike count.
  3. Execute each layer in its cheaper mode.
  4. Evaluate on GPU (measured) and on neuromorphic hardware via published cost models (Loihi 2 runtime model, Floorline).
- **Key premise (pilot):** in standard trained SNNs (e.g., on SHD), per-layer spike rates are spread *around* the crossover, with some layers below and some above. If all layers sit far on one side, there is nothing to allocate and N1 dies.
- **Nearby work (memory, unverified):** Floorline's sparsity-aware training plus partitioning (closest threat); layer-wise spike regularization; hardware-aware SNN NAS; NeuroBench metrics.
- **Collision risk:** MEDIUM–HIGH. Real-life value: SNN keyword spotting and gesture recognition on edge GPUs and neuromorphic chips.

## N2. Barrier-free, exact event-driven SNN inference on GPUs

- **Problem:** GPU SNN frameworks (snnTorch, SpikingJelly, GeNN) advance all neurons in lock-step time steps. At low spike rates each step does tiny work, so per-step launch and synchronization overhead dominates. That is exactly the regime where barrier removal gave 25–163× in EXP-005. Timestep grouping (Spike simulator) only helps up to the minimum synaptic delay.
- **Neuromorphic core:** asynchronous, clock-free spike propagation, as in DYNAP and Speck and as targeted by NeuroScale in hardware, brought to commodity GPUs while staying **exactly equivalent** to clock-driven semantics.
- **Mechanism:** a persistent GPU kernel in which neuron partitions advance their own local time independently. Each partition waits only on the input partitions that can still send it a spike before its next time, using conservative lookahead from synaptic delays and refractoriness. No global time step.
- **Key premise (pilot):** in trained SNNs at realistic spike rates, per-time-step GPU time is dominated by fixed overhead rather than spike work, by a large factor. Measure with an existing framework; if per-step work dominates, N2 dies.
- **Nearby work (memory, unverified):** Spike simulator (timestep grouping), GeNN, NEST GPU, GPU parallel discrete-event simulation (Time Warp on CUDA), NeuroScale (local sync in hardware).
- **Collision risk:** MEDIUM–HIGH.

## N3. Truly event-driven recurrent processing for event cameras (per-event state updates)

- **Problem:** most event-camera deep models bin events into frames or voxel grids, which throws away the microsecond timing and the sparsity that make event sensors useful. Real uses (AR/VR eye tracking, drones) need low latency.
- **Neuromorphic core:** stateful spiking or recurrent neurons updated *per event*, only where events land, with exact continuous-time leak between updates.
- **Key premise (pilot):** per-event updates touch a small fraction of the state, so latency per event beats per-frame processing at equal accuracy.
- **Nearby work (memory):** event-based eye-tracking challenges, asynchronous graph networks for events (AEGNN), event-based SSMs, EvRNNs.
- **Collision risk:** HIGH. Datasets are large (disk is a problem).

---

## Ranking

| | Neuromorphic depth | Premise pilot cost | Real-life | Novelty (unverified) | Disk fit |
|---|---|---|---|---|---|
| **N1 spike budgets** | high (spike-rate homeostasis, deployment cost) | ~1–2 h (SHD ~0.2–0.5 GB + snnTorch) | edge KWS/gesture, neuromorphic deployment | medium (Floorline threat) | yes |
| **N2 barrier-free GPU SNN** | high (clock-free spiking execution) | ~1 h | faster SNN deployment/simulation on GPUs | medium (Spike/GeNN threat) | yes |
| N3 event-camera per-event | high | 2–4 h | AR/VR, drones | low | weak (large datasets) |

**N1 and N2 are complementary:** N1 sets *how much* activity each layer should have; N2 makes low activity *pay off* on GPUs. Together they could form one paper: "make spiking sparsity pay off on real hardware".

## Literature check needed (requires user permission)

Before any pilot result is interpreted as novel: about **8–10 targeted web searches**, no agents, roughly 15–25k tokens. Topics:
- Floorline's training method;
- hardware-aware or layer-wise SNN spike regularization;
- deployment-cost-aware SNN training;
- event-driven, barrier-free GPU SNN simulation;
- Spike / GeNN / NEST GPU synchronization;
- GPU Time Warp for SNNs.

---

## Literature check (2026-10-04, user-permitted; 8 web searches, no agents)

### N1 (hardware-calibrated spike budgets) — DROPPED: high collision
- **Floorline** (Yik et al., [arXiv:2511.21549](https://arxiv.org/abs/2511.21549)): a two-stage procedure. *Sparsity-aware training* lowers the maximum per-neurocore synaptic operations at iso-accuracy, then *floorline-informed partitioning* follows. Up to 4.29× runtime and 4.36× energy on real neuromorphic chips. This already is "hardware-guided sparsity training", and on real hardware.
- **Feedback-control optimizer for hardware-aware SNN training** with target firing rates ([arXiv:2602.13261](https://arxiv.org/html/2602.13261v1), 2026).
- Layer and population firing-rate regularization is standard (snnTorch guide; sparsity-regularized backprop, [PMC9047717](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9047717/)).

### N2 (barrier-free exact event-driven SNN on GPU) — generic version: MEDIUM–HIGH collision
- **PDES for SNNs exists on CPU:**
  - Time Warp/optimistic SNN simulation (Pellegrini et al., [High Performance Simulation of SNNs](https://alessandropellegrini.it/publications/tPimp20.html)), which reports that conservative was faster than optimistic in most cases.
  - Spintronic SNNs evaluated with PDES ([ACM TOMACS 2024](https://dl.acm.org/doi/full/10.1145/3649464)).
- **GPU per-step overhead is a known problem, with partial fixes:**
  - timestep grouping up to the minimum delay ([Spike simulator](https://www.biorxiv.org/content/10.1101/461160v2.full));
  - multi-timestep kernel fusion of neuron dynamics ([SpikingJelly](https://www.science.org/doi/10.1126/sciadv.adi1480), up to 11× at T = 32);
  - temporal fusion for GPU SNN training ([ICANN 2024, arXiv:2408.00280](https://arxiv.org/html/2408.00280v1));
  - [Brian2CUDA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9660315/) and [Brian2GeNN](https://www.nature.com/articles/s41598-019-54957-7) are clock-driven per step;
  - a community project notes that GPU "kernel launch and synchronization dominate regardless of how few spikes" ([fly-brain](https://github.com/latteine1217/fly-brain)).
- So "GPU PDES applied to SNNs" alone would likely be judged incremental.

### N2* (refinement) — SURVIVES, unverified beyond 8 searches
- **Idea:** conservative lookahead derived from **neuron dynamics**, not synaptic delays. A partition whose LIF neurons are far enough below threshold *provably cannot spike for k steps*, even under maximal possible input. Its downstream partitions may therefore advance k steps without synchronizing. This works even with delay = 1, where timestep grouping fails.
- **Recursive version:** the input bound counts only presynaptic partitions that can themselves still fire (a fixed point, like null messages but computed from membrane state).
- **Closest found:** Lytton & Hines adaptive per-neuron integration steps that shrink near threshold (review: [Brette et al. 2007, PMC2638500](https://pmc.ncbi.nlm.nih.gov/articles/PMC2638500/)). That is an *integration-accuracy* technique, not cross-partition synchronization lookahead.
- **Neuromorphic depth:** high. It uses the defining property of spiking neurons (threshold, leak, refractoriness) to remove the global clock, which is what asynchronous neuromorphic chips do in hardware.
- **Main technical risk:** worst-case input bounds may be too loose (e.g., the sum of all positive incoming weights exceeds the threshold gap), giving a horizon of 1 almost always. The recursive bound exists to tighten this. **The premise pilot must measure the horizons.**

## PILOT-003 — N2* premise: provable silence horizons (PRE-REGISTERED 2026-10-05, before running)

- **Model:** recurrent LIF SNN, 700 inputs → 256 recurrent LIF → 20 leaky readout. SHD (Spiking Heidelberg Digits), 100 bins of 10 ms. β = exp(−10/20), θ = 1, reset by subtraction, surrogate-gradient BPTT on CPU, ~15 epochs, mild firing-rate regularization (standard practice). The trained accuracy is reported; if it is below 60%, the statistics are flagged as possibly unrepresentative.
- **Analysis** (test set, ~500 samples), for each neuron i, step t and window K ∈ {2, 4, 8}: neuron i is **provably silent for K** if, for every k ≤ K,
  β^k·V_i[t] + Σ_{m=1..k} β^{k−m} (I_ext_i[t+m] + R_i) < θ,
  where:
  - external input I_ext is known exactly (buffered input events);
  - recurrent input at m = 1 is exact (spikes at t are known);
  - for m ≥ 2 it is bounded by R_i;
  - **naive bound:** R_i = Σ_k max(W_rec[i,k], 0);
  - **recursive bound:** R_i counts only presynaptic neurons in the "may fire" set S. S is iterated from all neurons down to the greatest fixed point (sound by a first-firing-neuron argument).
- **Metrics:**
  - (a) fraction of neuron-steps provably silent;
  - (b) fraction of 32-neuron partition-steps provably silent;
  - (c) fraction of steps where the whole layer is provably silent (relevant for dense recurrence);
  - (d) actual silence (oracle), for tightness.
- **Criteria (recursive bound):**
  - **VIABLE:** K = 4 with (a) ≥ 50% **and** (b) ≥ 30%.
  - **KILL:** K = 2 with (a) < 10%.
  - Otherwise inconclusive.

### PILOT-003 — RESULT (2026-10-05): INCONCLUSIVE by the letter; NOT VIABLE in practice. N2* (conservative form) dropped.

- **Model:** RSNN 700 → 256 → 20 on SHD, 15 epochs on CPU. **Test accuracy 70.8%**; mean firing rate **2.08%** per neuron per step. Analysis on 500 test samples. File: `results/pilot003.json`.

| K | bound | neuron-steps provably silent | 32-neuron partition-steps | whole layer |
|---|---|---|---|---|
| 2 | naive | 11.2% | 0.0% | 0.0% |
| 2 | **recursive** | **25.1%** | **0.1%** | 0.0% |
| 2 | oracle (actually silent) | 97.1% | 65.5% | 56.8% |
| 4 | naive | 2.3% | 0.0% | 0.0% |
| 4 | **recursive** | **5.0%** | **0.0%** | 0.0% |
| 4 | oracle | 96.1% | 62.2% | 52.2% |
| 8 | recursive | 1.0% | 0.0% | 0.0% |
| 8 | oracle | 94.5% | 58.1% | 47.8% |

- **Scoring:**
  - KILL (K = 2, recursive neuron < 10%) not triggered (25.1%).
  - VIABLE (K = 4: neuron ≥ 50% and partition ≥ 30%) not met (5.0% / 0.0%).
  - Formally **inconclusive**.
  - But the decision-relevant quantity, partition-level provable silence, is ≈ 0 at every K, so the conservative mechanism has no headroom. **N2* in its conservative, exact form is dropped.**
  - Note: the recursive fixed point hit the 50-iteration cap at K = 2. Every iterate is a superset of the greatest fixed point, so the bound is still sound, just possibly not tightest.
- **What the data shows:**
  - The *opportunity* is real: the whole layer is actually silent in 52% of 4-step windows and 48% of 8-step windows.
  - Worst-case bounds (every positive presynaptic weight firing every step) certify none of it. The bound is too loose by orders of magnitude.
  - Exploiting real silence requires **optimistic execution with rollback**. That is Time Warp, already applied to SNNs on CPU (Pellegrini et al.; conservative was usually faster). On GPU, with p(silent window) ≈ 0.5 at K = 4, the rough best case is about a 2× reduction in synchronizations before rollback costs.
  - Verdict: modest headroom plus existing prior art. Not pursued without a new idea.
- **Caveats:**
  - Dense all-to-all recurrence is the worst case for such bounds. Sparse, local connectivity (cortical models, neuromorphic core mappings) would give tighter bounds. Untested.
  - One model, one seed.
- **Disk:** D: free fell to 2.1 GB (WSL disk growth from the dataset and model). Further downloads need care.

## PILOT-003b — N2* with LOCAL connectivity (PRE-REGISTERED 2026-10-05; one-shot follow-up, then N2* is closed either way)

- **Change from PILOT-003 (only this):** the recurrent weights are masked to local connectivity. 256 neurons in 8 partitions of 32 on a ring; each partition receives recurrent input only from itself and its two neighbours, as in core-local mappings on neuromorphic chips. Same data, training, seed and bounds.
- **Decision metric:** fraction of partition-steps in which **both neighbouring partitions** are provably silent for K steps (recursive bound). This is when a partition may advance K steps without synchronizing with others.
- **Criteria:**
  - **VIABLE:** K = 4 ≥ 30%.
  - **KILL:** K = 2 < 10%.
  - Otherwise inconclusive, treated as **not viable**: no further variants.
- Accuracy is reported. If local connectivity drops it below 60%, the result is flagged as unrepresentative.

### PILOT-003b — RESULT (2026-10-05): INCONCLUSIVE, so NOT VIABLE (as pre-registered). **N2* CLOSED.**

- Local ring connectivity (8 × 32, self + 2 neighbours). **Test accuracy 69.4%**, firing rate 1.91%. File: `results/pilot003b.json`.

| K | recursive: neuron / partition / **neighbours silent** | oracle: neuron / partition / neighbours |
|---|---|---|
| 2 | 96.9% / 67.6% / **62.0%** | 97.4% / 69.3% / 63.9% |
| 4 | 26.1% / 1.2% / **0.0%** | 96.4% / 65.3% / 60.6% |
| 8 | 18.4% / 0.2% / **0.0%** | 95.1% / 60.5% / 56.6% |

(The naive bound is ≈ 0 at partition level for every K.)

- **Scoring:** KILL not triggered (K = 2: 62.0% ≥ 10%). VIABLE not met (K = 4: 0.0% < 30%). This is inconclusive, which the pre-registration said to treat as not viable, with no further variants.
- **Insight worth keeping:**
  - With local connectivity, the recursive dynamics-based bound is **nearly tight for a 2-step horizon** (62.0% proven vs 63.9% actual).
  - It collapses beyond that, because worst-case recurrent input compounds per step.
  - The maximum benefit is therefore skipping every other synchronization (< 2×). Too small against strong baselines.
- **Status of all candidates so far:**
  - Measurement study: trap.
  - Streaming Whisper: premise false.
  - Edit-KV: not neuromorphic.
  - N1: prior art (Floorline).
  - N2 generic: prior art.
  - N2*: premise insufficient (dense: ~0; local: 2-step only).

## N3 — Certifiably silent SNNs (training for provable silence)

- **Idea:** borrow certified training (robustness via bound-based losses) and apply it to spiking dynamics. Train the SNN with a loss that pushes each neuron's **worst-case K-step membrane reach** below threshold *where the neuron is actually silent*. Quiet periods then become **provably** quiet, and synchronization can be skipped exactly. Motivated by PILOT-003/003b: real silence is ~60%, but provable silence is ~0% beyond K = 2.
- **Novelty:** unverified. A literature check follows only if the premise passes.

## PILOT-004 — N3 premise (PRE-REGISTERED 2026-10-05, before running)

- **Setting:** identical to PILOT-003b (local ring connectivity, SHD, same seed, 15 epochs).
- **Added loss:** λ · mean over (b, t, i) of relu(max over k = 1..4 of reach_k − θ), computed only where the neuron is actually silent in (t, t+4]. Here reach_k = β^k V[t] + Σ_{m=1..k} β^{k−m} (I_ext[t+m] + R_i), with R_i = Σ_j max(W_rec[i,j], 0) over the local mask (the naive bound, differentiable).
- **Settings tried:** λ ∈ {0.1, 1.0}. **Both are reported**, no cherry-picking.
- **Metric:** K = 4 recursive "both neighbours provably silent" fraction (as in 003b), plus test accuracy.
- **Criteria:**
  - **VIABLE:** some λ reaches K = 4 neighbours-silent ≥ 30% **and** accuracy ≥ 66.4% (at most 3 points below 003b's 69.4%).
  - **KILL:** no λ reaches ≥ 10% at K = 4 within the accuracy limit.
  - Otherwise inconclusive: at most one more λ is allowed, chosen *before* seeing its result, then a decision.

### PILOT-004 — interim (λ = 0.1) and integrity check (PRE-REGISTERED before the check runs)

- **Interim λ = 0.1:** accuracy 67.2%; K = 4 recursive neighbours-silent 56.0% (oracle 60.8%); K = 8 50.3% (oracle 56.7%). Before certified training (003b) both were 0.0%. This meets the VIABLE criterion *on paper*. Treated as **unconfirmed** until the integrity check below.
- **Integrity check** (λ = 0.1 rerun with the model saved, same seed):
  1. **Soundness:** the number of K = 4 neuron-windows certified silent that actually spiked must be exactly **0**. Any violation means a bug, and the result is void.
  2. **Non-triviality:** test accuracy with recurrent weights zeroed. If it is within 1 point of the full model, the network does not use recurrence, and "provable silence" would be trivial (an effectively feedforward SNN, where multi-step processing without synchronization is already known). The result would then be **void as a contribution**.
  3. **Reported:** accuracy with only excitatory recurrent weights zeroed; recurrent drive R_i against threshold θ.
- **λ = 1.0 (complete):** accuracy **66.3%**, 0.1 points below the 66.4% limit, so it does **not** qualify. K = 4 recursive neighbours-silent 58.6% (oracle 60.7%); K = 8 53.0% (oracle 56.2%). Stronger λ certifies slightly more at slightly lower accuracy, as expected. Only λ = 0.1 qualifies, pending the integrity check.
- **Disk:** D: free 1.65 GB.

### PILOT-004 — RESULT (2026-10-05): **VIABLE, and integrity checks PASSED**

- λ = 0.1 rerun reproduced the first run exactly (deterministic). Model saved at `~/research/models/pilot004_l0.1.pt`.
- **Soundness:** K = 4 neuron-windows certified silent but actually spiking = **0**.
- **Non-triviality:** test accuracy 67.2% full; **59.0% with recurrent weights zeroed** (−8.2 points, beyond the 1-point triviality margin); 63.3% with only excitatory recurrence zeroed. **Recurrence is used.**
- **Recurrent drive:** R_i = Σ relu(W_rec) has mean 0.338 (max 0.518) against θ = 1.0, while Σ|W_rec| has mean 2.20. Training kept excitatory drive bounded while total recurrence stays large (largely inhibitory). This resembles E/I balance: a hypothesis, not yet a finding (no pre-training comparison saved).
- **Certified fraction, both neighbours silent:**
  - K = 4: 56.0% recursive and **37.7% naive** (cheap, no fixed point), against 60.8% oracle.
  - K = 8: 50.3% recursive, against 56.7% oracle.
  - The recursive bound captures 92% of actual 4-step silence.
- **Pre-registered verdict: VIABLE** (≥ 30% at K = 4, accuracy within 3 points), and integrity holds.
- **Open caveats (must be addressed before any claim):**
  - one seed, one dataset (SHD), small network, low absolute accuracy (67% vs much higher SOTA on SHD);
  - certification ≠ speedup: a real execution engine that skips synchronization still has to be built and benchmarked against strong baselines;
  - the runtime cost of the certificate must be measured;
  - the E/I-balance observation is untested;
  - novelty is unverified (literature check next).

### N3 novelty check (2026-10-05; 3 web searches, no agents)

- **Closest prior work:** certified *adversarial-robustness* training for SNNs, S-IBP / S-CROWN ([arXiv:2205.01625](https://arxiv.org/pdf/2205.01625)). It bounds neuron behaviour under perturbed inputs. **The bounding technique is not our novelty; cite it.**
- **Also nearby:** early-cutoff regularization ([arXiv:2301.09522](https://arxiv.org/html/2301.09522)) and sparsity regularization ([PMC9047717](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9047717/)). Both reduce computation but give **no guarantee**.
- **Not found** (11 searches in total across checks): training objectives that certify **temporal silence under worst-case recurrent input** in order to make **synchronization-free execution exact**.
- **Novelty claim (provisional):** "certified silence" as a training objective for exact barrier-free spiking execution. Risk: MEDIUM. A reviewer may say "IBP applied to a new objective"; the defence must be the execution mechanism and its measured benefit.

## PILOT-005 — seed replication (PRE-REGISTERED 2026-10-05, before running)

- λ = 0.1, local connectivity, seeds {1, 2, 3}, otherwise identical to PILOT-004.
- **Criterion:** all 3 seeds give K = 4 recursive neighbours-silent ≥ 30%, accuracy within 3 points of each seed's own λ = 0 baseline, 0 soundness violations, and an accuracy drop > 1 point when recurrence is zeroed.
  - The λ = 0 baselines for seeds 1–3 are trained too (6 runs, ~75 min on CPU).
- If any seed fails, report the replication as **partial** and diagnose it. No claim rests on seed 0 alone.

### PILOT-005 — RESULT (2026-10-05): **REPLICATED, all 3 seeds pass all pre-registered checks**

| seed | acc λ=0 → λ=0.1 | K=4 neighbours-silent, recursive: λ=0 → λ=0.1 (oracle) | naive at λ=0.1 | violations | acc with recurrence zeroed (λ=0.1) | R_i mean λ=0 → λ=0.1 |
|---|---|---|---|---|---|---|
| 1 | 68.2 → 69.4 | 0.0 → **57.9%** (61.3) | 50.2% | 0 | 53.4 (−16.0) | 1.445 → 0.327 |
| 2 | 67.5 → 67.3 | 0.0 → **57.1%** (60.3) | 49.9% | 0 | 49.8 (−17.5) | 1.444 → 0.335 |
| 3 | 69.7 → 68.2 | 0.0 → **57.1%** (60.4) | 49.8% | 0 | 56.9 (−11.3) | 1.466 → 0.326 |

- **Accuracy change:** +1.2, −0.2, −1.5 points (mean −0.17). There is no meaningful accuracy cost at this scale.
- **Certified share of actual 4-step silence:** ~94–95%.
- **Mechanism, consistent across seeds:**
  - excitatory recurrent drive bound R_i goes from ~1.45 (above θ) to ~0.33 (below θ);
  - total |W_rec| goes from ~3.2 to ~2.2;
  - λ = 0 baselines also show 0 violations (the bound is sound; it just certifies nothing).
- **Still open:**
  - a strong-model regime (SOTA-level SHD accuracy);
  - a second dataset (disk: D: free 1.18 GB);
  - **a demonstrated reduction in synchronization and communication** (next: PILOT-006);
  - a formal test of the E/I-balance hypothesis;
  - novelty beyond 11 searches.

## PILOT-006 — does certified silence reduce synchronization traffic? (PRE-REGISTERED 2026-10-05, before running)

- **Models:** the saved PILOT-005 models (seeds 1–3; λ = 0 control and λ = 0.1). 500 SHD test samples, T = 100.
- **Abstraction:** 8 partitions of 32 neurons = 8 cores on a ring (the local mask). A core must know its neighbours' spikes (or their absence) before advancing.
- **Each core's certificate is LOCAL and sound:**
  - c_P(t) = the largest k ≤ 16 such that every neuron in P provably cannot reach θ within k steps;
  - uses exact external input and the exact recurrent input at the first step, then the worst case R_i = Σ relu(W_rec · mask) from *all* presynaptic neurons;
  - needs no global information.
- **Messages per core per neighbour link, over T steps:**
  - **Lock-step:** T (one spike-or-null message every step).
  - **Certified:**
    - if the core spikes at step t: 1 message, then advance 1 step;
    - otherwise, if c_P(t) ≥ 1: 1 certificate message covering c steps, then advance c steps;
    - otherwise: 1 null message, then advance 1 step.
  - **Oracle (lower bound):** 1 message per spike step, plus 1 message per maximal silent run.
- **Metric:** reduction factor = lock-step messages / certified messages, averaged over cores, samples and seeds.
- **Criteria:**
  - **VIABLE:** λ = 0.1 mean reduction ≥ 2.0× (strong if ≥ 3.0×), **and** the λ = 0 control stays < 1.3× (showing the gain comes from the training).
  - **KILL:** λ = 0.1 < 1.5×.
- **Not measured here:** wall-clock time. That needs an execution engine (PILOT-007) and is only worth building if this passes.

### PILOT-006 — RESULT (2026-10-05): **INCONCLUSIVE** (control condition failed). Benefit real but modest.

| | λ = 0 (control) | λ = 0.1 (certified) | oracle |
|---|---|---|---|
| message reduction vs lock-step (mean ± sd, 3 seeds) | **1.52 ± 0.02×** | **2.17 ± 0.05×** | 3.00–3.03× |
| mean certified horizon when ≥ 1 | 1.00 | 4.02 | — |
| fraction of core-steps with horizon ≥ 1 | 0.705 | 0.708 | — |

- **Scoring:**
  - λ = 0.1 ≥ 2.0×: met.
  - Control < 1.3×: **FAILED** (1.52×).
  - KILL (< 1.5×): not triggered.
  - So **inconclusive**.
- **Why the control fails:** untrained networks already admit **1-step** certificates in ~70% of core-steps (consistent with PILOT-003b, where K = 2 was nearly tight), so each certificate covers 2 steps and messages roughly halve during silent periods. This was not anticipated when the criterion was set; it is recorded as a mis-specified expectation, not explained away.
- **Gain attributable to certified training:** 2.17 / 1.52 ≈ **1.43×**.
- **Ceiling:** the oracle is only ~3×, because each 32-neuron core spikes in ~30% of steps, and spike messages are unavoidable. Certified training reaches **72% of the attainable reduction**.
- **Assessment:**
  - The mechanism works and is sound, but in this regime its *system-level* benefit is modest: ~1.4× from training, ~2.2× in total, with a hard ceiling near 3×.
  - Larger gains would require regimes with much sparser per-core activity (higher ceiling), e.g. lower firing rates, or cores whose neurons fire rarely. That is untested.
  - Wall-clock benefit is unknown, and only matters where synchronization dominates (multi-core chips, distributed simulators).

## PILOT-007 — sparse-regime test: does certified training pay off more when activity is sparser? (PRE-REGISTERED 2026-10-05, before running)

- **Rationale:** PILOT-006 showed a ceiling of ~3× because 32-neuron cores spike in ~30% of steps. Sparser per-core activity raises the ceiling.
- **Variants** (seed 1 screening; everything else as in PILOT-005):
  - **V1:** firing-rate target 1% (penalty relu(rate − 0.01)) with 8 cores × 32.
  - **V2:** firing-rate target 1% with 16 cores × 16 (ring-local: self + 2 neighbouring cores).
  - Each is trained at λ = 0 (control) and λ = 0.1 (certified): 4 runs.
- **Metric:** training gain = (lock-step / certified messages at λ = 0.1) ÷ (lock-step / certified messages at λ = 0), using the PILOT-006 protocol (local certificates, KMAX = 16).
- **Decision rule:**
  - **CONTINUE** (replicate seeds 2–3, then build the execution engine): some variant reaches training gain ≥ 2.0 **and** λ = 0.1 accuracy within 3 points of its own λ = 0 control.
  - **STOP:** otherwise. Write N3 up as a modest short paper (certified silence: sound, ~1.4× synchronization reduction attributable to training, ceiling analysis). No further regime search.

### PILOT-007 — RESULT (2026-10-05): rule says **STOP**. N3 to be written up as a modest short paper.

| variant (seed 1) | acc λ=0 → λ=0.1 | msg reduction λ=0 → λ=0.1 | **training gain** | oracle ceiling | firing rate λ=0 / λ=0.1 |
|---|---|---|---|---|---|
| V1: 8×32, rate target 1% | 71.4 → 70.6 | 1.544× → 2.220× | **1.44×** | 3.16–3.18× | 2.43% / 2.41% |
| V2: 16×16, rate target 1% | 67.3 → 69.4 | 1.626× → 2.558× | **1.57×** | 3.77× | 2.67% / 2.61% |

- 0 soundness violations in all 4 runs; recurrence is used in all of them (accuracy drops when zeroed). V2 at λ = 0 has a smaller drop (67.3 → 62.3).
- **Decision rule:** CONTINUE required a training gain ≥ 2.0. Neither variant reaches it (1.44×, 1.57×). **STOP.**
- **Manipulation-check failure, stated openly:**
  - the 1% rate target did **not** lower firing rates (still 2.4–2.7%; the penalty weight of 1 was too weak);
  - so the "sparser activity" arm was effectively untested, and only core size changed;
  - per the pre-registered rule there is **no** rerun with a stronger penalty (no regime fishing);
  - this is listed as a limitation and as future work.
- **Trend:** smaller cores raise both the ceiling (3.2 → 3.8×) and the training gain (1.44 → 1.57×). That suggests the benefit grows with finer partitioning or sparser cores, but it is unconfirmed.

## N3 — FINAL STATUS (2026-10-05): write up as a short paper; stop investing in system-level claims

**Supported claims** (pre-registered, 3–4 seeds):
1. A certified-silence training loss makes recurrent SNNs provably silent over multi-step horizons: K = 4 neighbour silence goes from 0% to 57–58%, against ~61% actual. That captures ~94% of real silence.
2. The certificates are sound (0 violations across 10 trained models).
3. There is no meaningful accuracy cost (mean −0.2 points over 3 seeds), and recurrence remains functionally important.
4. Mechanism: excitatory recurrent drive bound goes from ~1.45 to ~0.33 of threshold, with total recurrence largely inhibitory.
5. System level: 2.2–2.6× fewer synchronization messages than lock-step. ~1.4–1.6× of that is attributable to training (untrained networks get ~1.5–1.6× from 1-step certificates). The oracle ceiling is 3.2–3.8× in this regime.

**Not supported or untested:** wall-clock speedup; strong-model (SOTA-accuracy) regime; other datasets; genuinely sparser activity regimes; real neuromorphic hardware.
