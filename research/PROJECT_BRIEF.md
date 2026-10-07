# Project Brief (Layer 2)

Last updated: 2026-10-06

## STATUS 2026-10-06 17:00 — six-seed result: the accuracy cost is ~0.9 points, not ~0

**CONFIRM-002b (seeds 5-7 added, all six reported as pre-registered).** The 12:45 claim of "no measurable
cost" is **WITHDRAWN**.

| H=512 SHD, test | seeds 2-4 | **all six seeds (2-7)** |
|---|---|---|
| mean cost | -0.43 | **-0.91** (sd 0.96, se 0.39, 95% CI **[-1.68, -0.14]**) |
| mean certified | 55.98% | **55.77%** (range 53.94-57.77) |
| violations | 0 | **0** |

All three added seeds were worse (-0.88, -1.72, -1.59). **The 95% CI now excludes zero**, so there is a
real, measurable cost of about **0.9 points (CI 0.1-1.7)**. The pre-registered <= 1.0 bar is met by the
point estimate, but **the bar lies inside the CI**, so a cost above 1.0 cannot be excluded.

**Also revised:** fine-tuning's advantage over from-scratch is smaller than first measured -- -0.91 (FT,
6 seeds) vs -1.09 (scratch, 3 seeds). A 6-seed scratch arm is needed before claiming FT is meaningfully
better on *accuracy* at this width; the *certifiability* gap stays clear (55.8% vs 60.5%).

**Methodological note:** the 3-seed estimate was optimistic by ~0.5 points -- more than the whole
remaining margin to the bar. The anti-optional-stopping commitment (report all seeds, no reselection) is
what prevented the favourable 3-seed figure from reaching the manuscript.

**Infrastructure escalation:** GPU concurrency is unreliable on this WSL/Blackwell setup at **any** level,
not just 6 processes. At 2-3 processes we saw `CUDA_ERROR_UNKNOWN` *and* a bogus OOM (48 MiB refused with
10.45 GiB free, while WSL reported "17179869184 GiB in use" = 2^34 -- corrupted GPU memory accounting).
**All runs are now serial** (`run_serial_pending.sh`); since each failure costs a full rerun, serial is
faster in expectation. Long jobs are launched detached (`setsid nohup`) because session restarts twice
killed running work.

## TITLE v5 (adopted 2026-10-07) and NCE manuscript drafted

**Title v5:** *Short-Delay Excitatory Drive Governs Provable Silence in Recurrent Spiking Networks*

Why revised from v4 (*Certifiably Silent Spiking Networks: Bounding Excitation Makes Recurrent SNN
Silence Provable and Reduces Synchronization in Exact Multi-Core Execution*):
- **"Bounding Excitation" was wrong.** DCLS-001 showed the *unconstrained* learnable-delay model
  certifies 55.88% at 97.4% of oracle. The governing factor is not how much excitation there is but
  **which delays carry it** -- R_short, the drive through synapses with delay below the horizon.
- **"Reduces Synchronization" overclaimed.** That is a shared-memory many-core result (1.25-1.49x at
  32 cores); the real two-process TCP configuration was **1.37x slower**. It cannot sit in the title.
- v5 names the governing quantity, which is the paper's actual contribution and what the NCE
  readership will judge.

**Target venue: Neuromorphic Computing and Engineering (IOP Publishing).** Chosen over Neural Networks
/ Neurocomputing because that readership judges the execution contribution rather than benchmark
accuracy, treats emulated and simulated execution as normal, and values exactness guarantees. Quartile
to be verified on Scimago/JCR before submission.

**Manuscript drafted:** `research/paper/manuscript_nce.md` (source), `manuscript_nce.docx` (built),
`manuscript_nce.tex` (iopart format). **The IOP template is NOT on this machine** -- only IEEEtran.cls
and an Elsevier kit were found, and no TeX distribution is installed, so the .tex has **not been
compiled**. It needs `iopart.cls` from IOP's author site, or Overleaf's IOP template. Placeholders
remain for authors, affiliation, email, acknowledgements, repository URL and the AI-use declaration,
and twelve references are marked [verify].

## STATUS 2026-10-07 05:40 — CONFIRMED on test with fresh seeds AND out-of-sample. Morning summary: `research/paper/README_MORNING_2.md`

**All overnight work complete.** The 03:20 status below was seed-1 validation; it is now confirmed.

| | SHD (test, frozen, seeds 2-4) | SSC (test, frozen, no retuning) |
|---|---|---|
| control certified | 4.2% | 9.50% |
| **constrained certified** | **55.98%** | **60.90%** |
| % of oracle | **95.0%** | **94.3%** |
| **accuracy cost** | **+0.24** (sd 0.38, CI ~[-0.19,+0.67]) | **+3.10** |
| test accuracy | **87.59%** | 68.66% |
| violations | **0** | **0** |

**Validation-selection bias proved negligible** -- the risk I flagged hardest. Seed-1 validation predicted
+0.60 / 56.28% / 94.9%; fresh-seed test delivered **+0.24 / 55.98% / 95.0%**. `R_short` lands at
0.253/0.253/0.260 across the three seeds: the mechanism reproduces.

**SPEED (idle CPU, handshake given the delays' free lookahead, exact=1 everywhere):** certificates win
**only at 32 cores, 1.25-1.49x**, and **lose at 4-16 cores at low latency** (0.78-0.95x) because their
runtime cost exceeds their benefit once d_min=2 hands the baseline a free step. The control never wins, so
the 32-core gain is entirely training-attributable. This is **lower than the single-delay architecture's
1.67-1.71x**: delays buy accuracy and certifiability at the cost of the certificate's marginal speed
advantage. Pre-registered as the expected outcome before measuring. Caveat: 32 cores on 32 threads
saturates the scheduler and plausibly inflates the benefit of waiting less.

**Also settled:** 300 epochs is WORSE than 150 on this architecture (87.25 -> 84.69 control), unlike the
delay-free model. The frozen 150-epoch recipe is correct.

**Remaining gaps:** no real hardware (emulated latency); sub-SOTA accuracy (87.59 vs ~96 SHD, 68.66 vs ~80
SSC -- the fix is per-synapse learnable delays, since multi-tap multiplies parameters and the model is
data-limited at ~1M); SSC is seed 1 only; the speed story is now architecture-dependent and each number
must be attributed to its architecture.

## STATUS 2026-10-07 03:20 — certification is FREE on two datasets; the design rule was DERIVED, not fitted

**The project's central problem is solved.** Provable silence now costs **nothing** -- it *gains* accuracy --
and this has been shown on two datasets with a frozen recipe and zero retuning between them.

| | SHD (val) | SSC (test, frozen, no retuning) |
|---|---|---|
| control certified | 3.54% | 9.50% |
| **constrained certified** | **56.28%** | **60.90%** |
| % of oracle | **94.9%** | **94.3%** |
| **accuracy cost** | **+0.60** | **+3.10** |
| violations | 0 | 0 |

### What changed: synaptic delays, and a decomposition that was derived before it was measured

With v(t+1) = BETA*v(t) + I(t+1) + sum_d W_d s(t+1-d), a certificate at horizon k needs spikes from
t+1+k-d. **For d >= k those are already past, so they are known exactly; only d < k must be bounded.**
Therefore only taps with d < K bind a K-step certificate, and the network can satisfy the budget by
**moving** excitation into the free long taps instead of destroying it.

That prediction came from the algebra, before any run. Both datasets then showed exactly it:

| R_per_delay | d=2 | d=4 | d=8 | total |
|---|---|---|---|---|
| SHD control | 4.97 | 4.90 | 5.40 | 15.27 |
| SHD constrained | **0.28** | 6.03 | 6.46 | 12.76 (-16%) |
| SSC control | 5.73 | 6.12 | 7.19 | 19.04 |
| SSC constrained | **0.29** | 8.56 | 9.98 | **18.83 (-1%)** |

The binding d=2 tap collapses to ~0.28 on **both** datasets -- just under the 0.3935 budget -- while the
free taps grow and total excitation is nearly unchanged. **The constraint became a reallocation, not a
reduction.** Contrast the old single-delay architecture, where the same budget forced destroying 92% of
recurrent excitation (E/I 0.650 -> 0.045) and cost accuracy. **Delays give the network somewhere to put
its excitation.**

**Design rule for the paper:** use delays >= 2 and omit the unit-delay tap. It improves accuracy *and*
certifiability simultaneously -- there is no trade-off on this architecture.

### Accuracy also moved a long way
Base model 77.42 -> **87.25** validation on SHD (~90% test) from stronger augmentation + longer schedules
+ delays. The faster-leak and local-connectivity levers (BUDGET-001, WIDTH-001, RADIUS-001) are
**superseded** -- once delays do the work, the budget stops being the binding constraint.

### Established and robust
- **0 soundness violations** across every run, dataset, width and seed to date.
- The delay-aware exact engine is **bit-identical to a single-thread reference in 80/80 configurations**
  (4/8/16/32 cores x 5 latencies x 2 modes x 2 models).
- Certificate coverage **rises with core count** (43.8% -> 71.5% for 4 -> 32 cores); the unconstrained
  control gets 0.8-1.2%, so **training supplies essentially the entire effect**.
- `R ~ 0.0098 x fan-in`, linear across a 10x range: a predictive design formula.

### NOT yet established — do not quote these as settled
1. **Seeds.** Everything above is seed 1. SHD-CONFIRM-FROZEN (test set, seeds 2-4) is running.
2. **Wall-clock speed-up on the delay architecture is UNKNOWN.** The 1.67-1.71x figure belongs to the
   *single-delay* model. Two attempts to measure the delay engine were contaminated by concurrent jobs
   and discarded. Expectation recorded in advance: certificates should gain **less** here, because
   d_min = 2 already hands the baseline a free step; if that free lookahead captures most of the benefit,
   it is a negative result for the certificate contribution and will be reported as one.
3. **No real hardware.** Latency is still emulated.
4. **Sub-SOTA accuracy.** ~90% test on SHD vs ~96%; 68.66% on SSC vs ~80%.

## STATUS 2026-10-06 13:30 — the fine-tuning fix does NOT generalize to 1,024 neurons

**SCALE-FT-001 FAIL.** Open item 1 of the 12:45 status below is now answered, negatively. The
CONFIRM-002 recipe must be **scoped to H = 512 on SHD**.

| H = 1024, SHD, seeds 1-3, test | acc | cost | certified | R_mean (budget 0.3935) |
|---|---|---|---|---|
| control       | 79.77 |  --   |  0.00% | 8.72 |
| ours (FT)     | 79.30 | -0.47 | **0.40%** | **0.510** |
| ref (scratch) | 77.00 | -2.77 | 59.99% | 0.392 |

**Fine-tuning kept the accuracy by not actually satisfying the constraint.** R_mean stalled at 0.510,
*above* the budget. Certification is a threshold in R vs budget (Proposition 1), so 30% over the budget
certifies ~0% rather than proportionally less. The mechanism is confirmed even though the experiment failed.

**Consequences:**
- The manuscript's **"accuracy cost grows with size" limitation STANDS** and stays in the paper.
- At H = 1024 the trade-off is currently **binary**: accuracy (FT, no certificates) **or** certificates
  (scratch, -2.77 points). No measured setting gives both at this width.
- CONFIRM-002 (-0.43 at no measurable cost) is a **512-on-SHD** result, not a general one.

**Mechanistic cause -- the size effect is a fan-in effect.** Control R_mean rises 4.78 (H=512) -> **8.72**
(H=1024), i.e. 1.83x for 2x width, while the budget (1-beta)*theta = 0.3935 is fixed. Worst-case
excitatory drive grows roughly linearly with fan-in, so the constraint tightens proportionally as the
network widens, and a fixed-length fine-tune cannot cover the larger distance.
**Correct implication (an earlier note here overstated it as "fan-in-aware budgets"):** the budget is
fixed by the neuron model (theta, beta) and *cannot* be scaled with fan-in, and the scratch arm reaches
R = 0.392 < 0.3935 **at H = 1024**, so the budget is demonstrably reachable at that width. Feasibility is
not the problem -- the optimization path and its accuracy cost are. The sharper implication is:
**the method scales with bounded fan-in, not with width.** R_i is the sum of positive recurrent weights
into neuron i, so under *dense* recurrence fan-in = H and R grows with width; under *local* recurrence
fan-in is set by the neighbourhood and stays constant as H grows, so R -- and certifiability -- should be
width-independent. This predicts certification holds at any width under local connectivity, which is also
already the regime with the best measured speed-ups (1.5-2.3x local vs 1.15-1.40x dense). If it holds, the
paper's "degrades with size" limitation becomes a **scoping statement** ("scales under bounded fan-in,
which is the regime where it pays off most") rather than a defect.
**Untested:** `s4_improve.py` has no local-connectivity path (its mask only implements sink hubs); ring-local
connectivity exists only in `pilot_silence.py` (`LOCAL=1`, `local_mask()`). Testing this needs a small,
well-defined port of that mask into `s4_improve.py`.

**Claim-level upside:** the theory now predicts its own failure mode -- provability is governed by the
excitatory-drive budget, whose required margin scales with fan-in. That is a sharper and more defensible
claim than an empirically reported limitation.

**In flight:** CONFIRM-002b seeds 6-7 (seed 5 done; 6-7 were lost to a CUDA incident) and **SCALE-FT-002**
stage 1 -- a fine-tuning-budget sweep at H = 1024 ({20,40} epochs x lambda {0.3,1.0}), **validation only**,
to test whether the 1024 failure is merely an insufficient fine-tuning budget. Pre-registered, with no test
evaluation unless a config reaches the budget. See `N3_SCALEUP_PLAN.md`.

**Infrastructure:** max **3 concurrent CUDA processes** on this machine. Six caused `CUDA_ERROR_UNKNOWN`
and destroyed 5 runs while measurement showed 6 jobs give **no** throughput gain over 3 (launch-latency
bound workload; 140 W of a ~360 W budget). Batch size and precision deliberately unchanged -- both would
break comparability and precision affects the R_i bound arithmetic underlying soundness.

## STATUS 2026-10-06 12:45 — the accuracy problem is SOLVED and CONFIRMED (now on RTX 5080)

**CONFIRM-002 PASS (seeds 2-4, test set, one-shot).** The project's main open problem -- the accuracy
cost that sank ExCap at -4.5 points in CONFIRM-001 -- is resolved.

- **Final confirmed recipe:** fine-tune from that seed's trained control (20 epochs, lr 5e-4, constraint
  ramped over the first 10 epochs), cert lambda = 0.3. Selected on speaker-disjoint validation, confirmed
  on fresh seeds against the test set.
- **Result:** test cost **-0.43 points** (bar: <= 1.0), **55.98% certified** (bar: >= 55%, 89% of oracle),
  **0 violations**. Controls certify ~0%.
- **Core insight:** the cost came from *initialization*, not from the loss. The identical loss at identical
  strength costs -1.09 points from scratch and -0.43 as fine-tuning.
- **Trade-off knob to report as a pair:** scratch certifies more (60.45%, 95% of oracle) at -1.09;
  fine-tuning certifies less (55.98%, 89% of oracle) at -0.43.
- **Mechanism confirms Proposition 1:** mean R_mean lands at 0.3904 vs the 0.3935 budget.
- **Negative result:** knowledge distillation is counterproductive here -- it recovers accuracy (+0.09 above
  control) but pushes R above the budget, cutting certification to 53.7%.

**Honest limits:** with 3 seeds the per-seed costs span +0.88 to -1.41, so -0.43 is not statistically
distinguishable from zero *or* from a 1-2 point cost. Claim "no measurable cost at 512 on SHD", not "no cost".

**Open, in value order:**
1. **Does the fine-tuning fix generalize?** The documented cost growth at 1,024 neurons (-2.1) and on SSC
   (-3 to -3.7) was measured with *from-scratch* training and has not been retested with fine-tuning. If it
   generalizes, the strongest reviewer objection ("it degrades where it matters") largely dissolves.
2. More seeds, to tighten the accuracy claim (~4 min per seed on this machine).
3. **Retrain the final model on the full training set.** The speaker-disjoint split costs ~2.1 points of
   absolute accuracy (control 80.8% with 10 speakers vs 78.8% with 8); it is the right selection instrument
   but should not set the headline number. Costs are unaffected.
4. **Speed section decision:** the manuscript's 1.5-2.3x figures were measured on the 6-core i7-10750H.
   This machine has 32 threads, so numbers from the two machines must not be mixed in one table.
5. Learned synaptic delays -- potentially raises accuracy toward SOTA *and* adds exact lookahead.

**Harness flaw found and fixed:** runs were piped through `grep -E "^RESULT"`, discarding tracebacks, which
silently hid a failed `torch.save` and left stage 2 with no checkpoint (exit code 0, no errors). Stage 1,
stage 2 and CONFIRM-002 now log in full and fail loudly. **15 other scripts still use the risky form.**

**Environment:** WSL2 Ubuntu 26.04, Python 3.14, torch 2.14.1+cu130, g++ 15.2, RTX 5080 16 GB. 5 s/epoch
(the full 8-config sweep takes 13 min). Details and deviations: `N3_SCALEUP_PLAN.md` -> MACHINE MIGRATION.

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
