# N3 Short Paper — Figure Plan & Claim–Evidence Table (Phases 17 and 20) — v1, 2026-10-05

**Working title:** *Certifiably Silent Spiking Networks: Training Recurrent SNNs for Provably Synchronization-Free Execution*

**Target:** workshop / short paper (NICE, ICONS) or a short article in *Neuromorphic Computing and Engineering* / *Frontiers in Neuroscience*.
**Paper type:** a method paper with an honest system-level evaluation. **Not** a speedup paper.

## One-sentence contribution
A training loss that bounds each spiking neuron's worst-case multi-step membrane reach turns most of a recurrent SNN's natural silence into **provable** silence, without accuracy loss. This allows exact skipping of synchronization; we quantify how much (≈1.4–1.6× attributable to training, ceiling ≈3–4× in our regime).

## Figures (the argument must be visible from figures alone)

| # | Figure | Shows | Data |
|---|---|---|---|
| 1 | Concept | Lock-step SNN execution vs certified-silent execution: a core certified silent for k steps lets neighbours advance without waiting. The bound: reach_k = β^k·V + Σ β^{k−m}(I_ext + R). | schematic |
| 2 | **The gap** | Actual vs provable silence for untrained RSNNs (dense and local), K = 2, 4, 8: real silence ~60%, provable ~0% beyond K = 2. Motivates the method. | PILOT-003, 003b |
| 3 | **Main result** | Provable neighbour silence at K = 4 / 8, control vs certified training, 3 seeds, with the oracle line. 0% → 57–58% (oracle ~61%). | PILOT-005 |
| 4 | Cost | Accuracy control vs certified per seed (mean −0.2 points), plus the recurrence-ablation bars (recurrence still matters). | PILOT-005 integrity |
| 5 | Mechanism | Distribution of R_i (worst-case excitatory drive) relative to θ: control ~1.45 vs certified ~0.33; total \|W_rec\| ~3.2 → ~2.2. | PILOT-005 (needs a histogram from saved models) |
| 6 | **System-level** | Synchronization messages vs lock-step for control / certified / oracle, 8×32 and 16×16 cores. Shows the 1.5× "free" one-step effect, the training gain, and the ceiling. | PILOT-006, 007 |

## Claim–evidence table

| Claim | Evidence | Fig | Stats | Risk |
|---|---|---|---|---|
| C1. Untrained RSNNs are mostly silent but almost never *provably* silent beyond 2 steps | PILOT-003/003b: K = 4 provable partition silence 0.0–1.2% vs ~62–65% actual | 2 | 1 seed + 3 control seeds (PILOT-005 λ = 0: 0.0%) | low |
| C2. Certified training makes 57–58% of 4-step neighbour windows provable (≈94% of actual) | PILOT-005, 3 seeds; seed 0 gives 56% | 3 | 3 seeds, sd < 0.5 points | low |
| C3. Certificates are sound | 0 violations across 10 models (λ = 0 and λ = 0.1) | text | exact count | low (but the proof must be stated in the paper) |
| C4. No meaningful accuracy cost | Δacc +1.2, −0.2, −1.5 (mean −0.17); λ = 1.0 costs more | 4 | 3 seeds; no significance claim | medium (low-accuracy regime, 67–71%) |
| C5. Recurrence remains functional after certified training | Accuracy with W_rec = 0: −11 to −17.5 points | 4 | 3 seeds | low |
| C6. Mechanism: excitatory drive pushed below threshold | R_i mean 1.45 → 0.33, all seeds | 5 | 3 seeds | medium (E/I-balance interpretation is a hypothesis; say so) |
| C7. 2.2–2.6× fewer synchronization messages; ~1.4–1.6× attributable to training; ceiling 3.2–3.8× | PILOT-006 (3 seeds), PILOT-007 (1 seed) | 6 | 3 seeds for 8×32; **1 seed for 16×16** | medium |

## Limitations (must appear in the paper)
1. No wall-clock speedup measured; message counts are a protocol-level proxy.
2. Small RSNN (256 neurons), one dataset (SHD), accuracy 67–71% (well below SOTA).
3. Untrained networks already gain ~1.5× from 1-step certificates. Our pre-registered control criterion failed (PILOT-006). Report it.
4. The sparser-activity arm failed its manipulation check (firing rate did not drop). That regime is untested.
5. Novelty checked with 11 web searches only. Closest prior work: S-IBP / S-CROWN (certified robustness for SNNs, same bounding family), Time Warp / PDES for SNNs, Spike-simulator timestep grouping, NeuroScale.

## Remaining work (no new experiments needed except figure data)
- [ ] Write the soundness proof (recursive bound, first-firing-neuron argument) as a short lemma.
- [ ] Generate the Figure 5 histograms and the Figure 2/3/6 plots from the saved JSONs and models.
- [ ] Before submission, a proper related-work search (more web searches; mention the token cost to the user).
- [ ] Draft (≤ 6 pages), then a hostile-reviewer pass (Phase 21).
