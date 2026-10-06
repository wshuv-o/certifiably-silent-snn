# Phase 5 — From Data to Mechanism Expectations — v1, 2026-10-04

Context: EXP-001…006 were flagged as the measurement trap. This document uses that data **only** to decide what a new mechanism must do, and which directions the data already rules out.

**Constraint (user, 2026-10-04):** software-only, runnable on a commodity GPU and CPU (this laptop); no special hardware; start small, do not scope the whole project up front.

## 1. What the data actually shows (headroom, not findings)

Regret = time of a simple rule / time of an oracle that always picks the faster mode (`harness/regret_analysis.py`, using EXP-004 and EXP-005 data).

| Decision | Rule | Geomean regret | Max regret |
|---|---|---|---|
| Sparse vs dense, CPU | Ligra threshold (s < 0.05) | **1.027** | 1.98 |
| Sparse vs dense, CPU | best static threshold (hindsight) | 1.002 | 1.13 |
| Sparse vs dense, GPU | Ligra threshold | **1.082** | 3.64 |
| Sparse vs dense, GPU | best static threshold (hindsight) | 1.056 | 2.73 |
| Sync vs async, Galois BFS | always-async | — | **1.32** (shallow graph only) |
| Sync vs async, Galois BFS | always-sync | — | 163 |

**Reading:**
- Existing simple rules are already within a few percent of the oracle on average.
- A smarter switching mechanism (cache-aware thresholds, adaptive sync/async) has single-digit-percent average headroom on these workloads.
- The only large effect is **per-round synchronization overhead when per-round work is tiny** (163×). Graph runtimes already remove it with asynchrony.

## 2. Directions ruled out (with evidence)

| Direction | Killed by |
|---|---|
| Adaptive / cache-aware sparse-dense switching | Data: Ligra rule regret 1.03 (CPU), 1.08 (GPU). Prior art: Ligra, Beamer DO-BFS, Gemini, SEP-Graph. |
| Adaptive sync/async switching for graph traversal | Data: always-async regret ≤ 1.32. Prior art: PowerSwitch, SEP-Graph. |
| SNN timestep grouping within the minimum synaptic delay | Prior art: Spike simulator (Ahmad et al. 2018) — [bioRxiv](https://www.biorxiv.org/content/10.1101/461160v2.full) |
| Data-dependent temporal aggregation for SNNs | Prior art: "Collapse or Preserve" (2026) — [arXiv:2603.13810](https://arxiv.org/pdf/2603.13810) |
| Lazy / closed-form neuron updates between events | Prior art: exact event-driven SNN simulation (Brette et al. 2007) |

## 3. What any mechanism must deliver (expectations set before design)

A candidate is only worth building if it can plausibly meet **all** of these:

1. **Targets measured headroom.** It attacks a cost our data shows is large: per-step synchronization or launch overhead under low per-step work, or something comparably large that we first measure. It must not target a cost that existing simple rules already handle within ~10%.
2. **Beats the strongest existing method, not a strawman.** ≥ 2× on the target regime against the best published software approach (e.g., GeNN or Spike-style timestep grouping for SNNs; Galois/GAPBS for graphs), on identical hardware.
3. **Preserves results.** Bit-identical, or tolerance-identical with a stated bound, to the standard clock-driven or synchronous semantics, verified automatically.
4. **Degrades gracefully.** ≤ 10% slower than the baseline outside the target regime (e.g., at high activity).
5. **Passes the brainstorm test.** No credible "already known / just benchmarking" objection after a targeted prior-art search.
6. **Fits the constraint.** Implementable and evaluable on this laptop (RTX 2060 6 GB, 6-core CPU).

## 4. Remaining candidate (not adopted; needs a decision)

**C1 — Speculative multi-timestep execution for recurrent SNNs with 1-step delays on GPU.**
- **Observation:** at low firing rates, GPU SNN simulation is dominated by per-step launch and synchronization overhead (Spike paper: 65–90% at small scale). Timestep grouping removes this only up to the minimum synaptic delay, and many trained recurrent SNNs use delay = 1, so grouping cannot apply.
- **Hypothesis:** advance k steps optimistically inside one persistent kernel, assuming no cross-block spikes arrive. Detect violations and roll back only the affected blocks (Time Warp principle). The expected cost is then ≈ (1 + rollback rate × k) × work, against k × launch overhead saved.
- **Expected evidence:**
  - ≥ 2× over per-step execution, and over grouping where grouping is inapplicable, at firing rates ≲ 1–5% per step.
  - Bit-identical spike trains.
  - k adapted online so that high-activity slowdown is ≤ 10%.
- **Novelty risk: MEDIUM–HIGH.**
  - GPU Time Warp already exists for agent-based simulation, including reversible RNG ([survey arXiv:1807.01014](https://arxiv.org/pdf/1807.01014)).
  - No SNN-specific GPU instance found in one search, but "known PDES technique applied to SNNs" may be judged incremental.
  - Needs a deeper prior-art check (Time Warp + SNN, NEST/GeNN optimistic modes, persistent-kernel SNN simulators) **before** any code.
- **Kill criteria:** prior art doing optimistic multi-step SNN execution on GPU; or a small pilot showing < 1.5× in the target regime.

## 5. Honest status

- No candidate has yet passed all six expectations.
- The data rules out the "smart switching" family.
- C1 is the only surviving idea, with real novelty risk.
- A genuinely strong mechanism may need an idea that is not derived from our graph-traversal data at all. Phase 1 brainstorming should stay open rather than force C1.
