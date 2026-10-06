# N3 Scale-Up Phase — toward a top-venue method paper (declared 2026-10-05)

**Why a new phase (not fishing):** PILOT-007's STOP applied to the *current small regime*. This phase asks different, pre-registered questions about scale, real payoff and theory. Results from the small regime stand as reported.

**Target:** a method paper (NeurIPS/ICLR; or Nature Machine Intelligence / Nature Communications if the system results are strong). The contribution is a **new training objective and execution guarantee**, never a measurement study.

## Workstreams and pre-registered goals

| # | Workstream | Goal (pass bar) | Needs |
|---|---|---|---|
| S4 | **Theory:** soundness lemmas (below) + a certifiability–expressivity result | Proofs written and checked; a proposition linking certified horizon K to an excitatory-drive bound | nothing (doing now) |
| S1 | **Wall-clock payoff:** multi-process "neuromorphic core" execution engine (one process per core, shared-memory spike/certificate exchange, exact) | Lock-step / certified-model ≥ 3× **and** control-model / certified-model ≥ 1.5×, both under the same certificate protocol, with bit-identical spikes | CPU only |
| S3 | **Bigger gains:** longer horizons (K = 8–16), genuinely sparse regimes (a rate penalty that *passes* its manipulation check), tighter bounds | Training gain in messages ≥ 2× in a regime whose firing rate is verified lower | CPU only |
| S2 | **Strong models:** adaptive LIF / delays, near-SOTA on SHD, plus SSC as a second dataset | Certified training costs ≤ 1 point at ≥ 85% SHD accuracy | **~15–20 GB free disk + GPU PyTorch** (user must free `D:\hf_cache`) |
| S5 | **Real hardware (optional, high value):** Loihi 2 via the Intel INRC programme | Measured reduction in synchronization time or energy on chip | **User applies to INRC** (outward-facing, user's decision) |

## S4 — Soundness (draft lemmas)

**Model:** V_i[t+1] = β·V_i[t] + I_i[t+1] + Σ_j W_ij·s_j[t]; s_i[t+1] = 1[V_i[t+1] ≥ θ]; reset by subtraction. External inputs I are known for the window. Define, for a set S of neurons:
- U_i^(1) = β·V_i[t] + I_i[t+1] + Σ_j W_ij·s_j[t] (exact);
- U_i^(k) = β·U_i^(k−1) + I_i[t+k] + R_i(S), with R_i(S) = Σ_{j∈S} max(W_ij, 0), for k ≥ 2.

**Lemma 1 (naive certificate, S = all neurons).** If U_i^(k) < θ for all k ≤ K, neuron i emits no spike in (t, t+K].
*Proof.* By induction on k, as long as i has not spiked: V_i[t+k] ≤ U_i^(k). The base case is exact. For the step, recurrent input Σ_j W_ij·s_j ≤ Σ_j max(W_ij, 0) because s_j ∈ {0, 1}, β > 0 preserves order, and no reset occurs before a spike. So V_i stays below θ throughout. ∎

**Lemma 2 (recursive certificate).** Let S_0 = all neurons and S_{n+1} = F(S_n) ∩ S_n, where F(S) = {i : ∃k ≤ K, U_i^(k)(S) ≥ θ}. For any n (converged or not), every neuron outside S_n is silent in (t, t+K].
*Proof.* Suppose some neuron outside S_n spikes, and let i be one that spikes **earliest**, at t+k. Input at step t+m (m ≥ 2) comes from spikes at t+m−1 < t+k, and before t+k no neuron outside S_n has spiked. So i's recurrent inputs up to t+k come only from neurons in S_n (plus the exactly known s[t]). Hence V_i[t+m] ≤ U_i^(m)(S_n) ≤ U_i^(m)(S_r) for all m ≤ k, where r is the iteration at which i was removed (R_i is monotone in S, and S_n ⊆ S_r). Removal means U_i^(m)(S_r) < θ for all m ≤ K, contradicting the spike. ∎
(This also covers the non-converged iteration cap used in PILOT-003, so those results are sound.)

**Proposition (sketch, to formalize).** If external drive is bounded by ā during a quiet window and R_i < (1 − β)·θ − ā, then U_i^(k) stays below θ for all k. Neuron i is then certifiable for **unbounded** K: the reach converges to (ā + R_i)/(1 − β) < θ. This links the trained quantity R_i directly to horizon length. In the trained models R_i ≈ 0.33 vs (1 − β)·θ ≈ 0.39, which explains why horizons of 4–16 become possible. **Testable prediction:** the certified horizon distribution should lengthen sharply as R_i falls below (1 − β)·θ − ā.

## Order of work
1. S4 proofs → done in draft above; formalize the proposition.
2. S1 engine (CPU) → the decisive test for "real payoff".
3. S3 (CPU).
4. S2 when disk is freed.
5. S5 only if the user applies.

## Honesty rules for this phase
- Every goal above is fixed **before** its experiment.
- Failures are reported.
- No regime fishing: each workstream gets one declared configuration sweep, chosen in advance.

## S1 — PRE-REGISTRATION (2026-10-05, before running)

- **Engine** (`harness/s1_engine.py`): 8 OS processes = 8 cores × 32 neurons (ring-local weights from the saved PILOT-005 seed-1 models). Shared-memory spike buffers, per-step publish timestamps, busy-wait. float64 arithmetic, identical per-core update in the engine and in the reference.
- **Modes:**
  - **barrier:** global multiprocessing.Barrier every step.
  - **handshake:** each core waits for both neighbours' step-t spikes.
  - **cert:** the same, but a neighbour's published certificate "silent through step u" satisfies the wait. The certificate uses the Lemma 1 bound with worst-case R for every step after the current one (exact information about neighbours' next spikes is unavailable), and is recomputed only when the previous certificate expires.
- **Emulated interconnect latency L ∈ {0, 5, 20} µs:** published data becomes visible to neighbours only L µs after its publish timestamp. This is an emulation of chip/cluster interconnects, labelled as such.
- **Data:** 200 SHD test samples, T = 100, models λ = 0 (control) and λ = 0.1 (certified), seed 1.
- **Exactness:** every engine run's spikes must be bit-identical to the single-process reference, or the run is void.
- **Pass (at L = 20 µs):**
  - (i) barrier-time(certified model) / cert-time(certified model) ≥ 3×;
  - (ii) cert-time(control model) / cert-time(certified model) ≥ 1.5×.
  - Also reported: handshake vs cert, and L = 0 and 5 µs.
- **Kill:** (ii) < 1.2× at L = 20 µs. That would mean training buys no real wall-clock benefit in this engine.

## S1 — RESULT (2026-10-05): **INCONCLUSIVE; no wall-clock win over a local handshake**

- **Exactness:** all 18 runs bit-identical to the reference (ALL EXACT = True).
- **Median ms per sample** (200 samples, T = 100, 8 processes):

| L (µs) | mode | control | certified |
|---|---|---|---|
| 0 | barrier / handshake / cert | 49.2 / 3.77 / 5.84 | 50.1 / 4.03 / 5.18 |
| 5 | barrier / handshake / cert | 50.7 / 4.19 / 6.39 | 48.2 / 4.04 / 5.34 |
| 20 | barrier / handshake / cert | 51.5 / 5.61 / 7.72 | 48.5 / 5.65 / **6.14** |

- **Scoring (L = 20 µs):**
  - (i) barrier / cert (certified) = 7.9×, so the bar is met. **But** Python's mp.Barrier is very slow (~0.5 ms per step), so this baseline is inflated and not a fair claim.
  - (ii) cert(control) / cert(certified) = **1.26×**: below the 1.5× pass bar, above the 1.2× kill line, so **inconclusive**.
- **Critical observation:** the cert protocol is **slower than a plain local handshake** for both models (certified: 6.14 vs 5.65 ms; control: 7.72 vs 5.61 ms).
  - The cause is certificate computation overhead: a Python loop of up to 16 numpy ops per expired certificate, against ~40 µs of per-step compute and only ≤ 20 µs of latency.
  - Certified training *reduces* that overhead relative to control (fewer recomputations, longer horizons). That is why (ii) is above 1.
- **Interpretation:** in a software emulation where compute per step is large relative to interconnect latency, local handshaking (as in NeuroScale) already captures most of the synchronization benefit, and our certificates do not pay. A benefit requires:
  - (a) near-free certificate evaluation, e.g. the O(1) threshold rule from the Proposition (R_i + ā < (1 − β)·θ ⇒ unbounded horizon) instead of a 16-step loop;
  - (b) regimes where latency ≫ per-step compute (inter-chip / inter-node links, or fast compiled cores).
- **Not run:** any configuration beyond the pre-registered one.
- **Next (S1b, must be pre-registered before running):** a compiled per-core update (Numba) plus an O(1) certificate, and a latency sweep {20, 100, 500} µs anchored to cited real-system latencies. Pass bars are fixed before running. If S1b also fails to beat handshake, the system-level claim is dropped from the paper.

## S2 — strong models (PRE-REGISTERED 2026-10-05, before running). Disk freed by user: D: 29 GB.

- **Model:** a single recurrent layer of 512 **adaptive** LIF neurons (ALIF: threshold θ + adaptation, decay ρ, jump 0.02 per spike), **dense** recurrence, leaky readout. SHD, 100 bins, time-shift and channel-jitter augmentation, AdamW with a cosine schedule, 40 epochs, GPU (RTX 2060).
- **Certification for ALIF:** the bound uses the *base* threshold θ. The adaptive threshold only rises above θ, so certificates stay sound. Recurrent worst case R_i over all presynaptic neurons (dense). Partitioned into 16 cores × 32 for the core-silence metric.
- **Arms:** λ = 0 (control), 0.1, 0.3. Seed 1 for screening; replicate on seeds 2–3 if it passes. All arms reported.
- **Metrics:** test accuracy; K = 4 core-provably-silent fraction (recursive bound) vs oracle; soundness violations (must be 0); accuracy with recurrence zeroed.
- **Pass:**
  - the control reaches **≥ 85%** test accuracy (strong-model regime);
  - **and** some λ costs **≤ 1.0 point**;
  - **and** it certifies **≥ 50% of the oracle** core-silent fraction at K = 4.
- **Partial:** control 80–85%, reported as a "moderate-strength regime". **Fail:** certification needs > 1 point of accuracy at every λ.

## S1b — compiled engine + O(1) certificate (PRE-REGISTERED 2026-10-05, before running)

- **Engine:** C++ with std::thread (8 cores × 32 neurons), atomics for publish/certificate, timestamps for emulated latency L ∈ {0, 5, 20, 100, 500} µs, busy-wait. Per-step compute is ~1–2 µs, as on real cores. Spikes must be bit-identical to a single-thread reference (same double arithmetic and order).
- **Certificate (O(1) per neuron, from the Proposition):** neuron i is silent until the first future step τ with I_i[τ] > (1 − β)·θ − R_i, provided V_i < θ now. The core certificate is the minimum over its neurons. "Next exceedance" indices are precomputed per sample (input is known in advance).
- **Modes:** handshake vs cert, for the control and certified models (PILOT-005 seed-1 models). **The global barrier is dropped as a headline baseline:** handshake is the fair, strong baseline.
- **Pass:**
  - at L ≥ 20 µs, cert(certified) is faster than handshake(certified) by **≥ 1.5×**;
  - **and** cert(control) / cert(certified) ≥ 1.3×.
  - Real-system latency anchors must be cited in the paper.
- **Kill:** cert never beats handshake by ≥ 1.2× at any L ≤ 100 µs. Then the system-level speed claim is dropped from the paper.

## S1b — RESULT (2026-10-05): **FAILED** (kill condition met), but uninformative about certificates

- All runs exact. cert never beats handshake (e.g. L = 20 µs, certified model: 2.64 vs 2.58 ms; L = 100: 10.65 vs 10.51; L = 500: 49.74 vs 49.79).
- **Diagnosis:** certificate coverage was **0.2%** of core-steps (control 0%).
  - The O(1) "unbounded-horizon" rule needs I_i[τ] ≤ (1 − β)·θ − R_i for **all 32 neurons** of a core. The certified model's R_i reaches 0.46 > 0.3935, so a single such neuron voids the core certificate.
  - The rule is a correct but far too strict sufficient condition. **The experiment therefore did not exercise certificates**: an implementation-validity failure analogous to PILOT-007's manipulation-check failure.
  - My design error: I chose the O(1) rule to cut overhead, but in compiled code the K-step loop costs well under 1 µs.
- Recorded as a failure.

## S1c — FINAL speed attempt (PRE-REGISTERED 2026-10-05, before running). If it fails, the speed claim is permanently dropped.

- **Identical** to S1b, except the certificate is the **Lemma 1 K-step bound** (KMAX = 16): worst-case R for every future step, starting from the post-step state, recomputed when expired or after a spike. Same models, samples and latencies.
- **Manipulation check (new):** certified-model cert coverage must be **≥ 30%** of core-steps, or the run is invalid and reported as such. There will be no S1d.
- **Pass:** at L ≥ 20 µs, handshake(certified) / cert(certified) ≥ **1.5×**, **and** cert(control) / cert(certified) ≥ **1.3×**.
- **Kill:** cert(certified) beats handshake by < 1.2× at every L ≤ 100 µs. Then the speed claim is dropped from the paper.

## S1c — RESULT (2026-10-05): **PASS** (pre-registered final speed attempt)

- All runs exact (bit-identical to the single-thread reference).
- **Manipulation check:** certified-model cert coverage **67.4%** (≥ 30%); control 0.2%.
- **Median ms per sample** (200 samples, 8 threads, compiled C++):

| L (µs) | certified: handshake / cert | **speedup** | control: handshake / cert |
|---|---|---|---|
| 0 | 0.474 / 0.552 | 0.86× | 0.554 / 0.569 |
| 5 | 1.059 / 0.784 | 1.35× | 1.083 / 1.101 |
| 20 | 2.643 / 1.758 | **1.50×** | 2.614 / 2.625 |
| 100 | 10.511 / 6.803 | **1.55×** | 10.542 / 10.555 |
| 500 | 49.845 / 32.071 | **1.55×** | 49.755 / 49.780 |

- **Scoring:**
  - handshake/cert (certified) ≥ 1.5× at L ≥ 20: **met** (1.503, 1.545, 1.554). At L = 20 it is marginal.
  - cert(control)/cert(certified) ≥ 1.3×: **met** (1.49, 1.55, 1.55).
  - **PASS.**
- **Key finding:**
  - In a realistic asynchronous engine a core cannot know its neighbours' next-step spikes, so the certificate must use the worst case from the first future step onward.
  - Under that constraint **untrained networks are essentially never certifiable (0.2%)**, against 67.4% for certified training.
  - This **resolves the PILOT-006 control issue**: the "free" ~1.5× in PILOT-006 came from assuming exact next-step knowledge, which is unavailable in practice. **The whole wall-clock gain is attributable to certified training.**
- **Limits:**
  - ~1.55× is real but moderate;
  - certificates cost time when there is no latency (0.86× at L = 0);
  - latency is emulated (cite real interconnect latencies);
  - one seed and one small model so far.

## S2 — RESULT (2026-10-06, seed 1): **PARTIAL** (moderate-strength regime); the method holds

| λ | test acc | acc, recurrence zeroed | K = 4 core-certified | oracle | % of oracle | violations | R_i mean | firing rate |
|---|---|---|---|---|---|---|---|---|
| 0 | **82.0%** | 12.3% | 0.0% | 59.5% | 0% | 0 | 4.81 | 4.37% |
| 0.1 | 81.0% (−1.02) | 8.3% | 58.4% | 62.5% | 93% | 0 | 0.401 | 4.46% |
| **0.3** | **81.3% (−0.66)** | 11.4% | **59.1%** | 62.4% | **95%** | 0 | 0.330 | 4.06% |

- **Scoring:** the control's 82.0% < 85%, so per the pre-registration this is a **moderate-strength regime**, not "strong". In that regime λ = 0.3 meets both method bars: cost ≤ 1.0 point (0.66) and ≥ 50% of oracle certified (95%). λ = 0.1 misses the cost bar by 0.02 points (single seed; within noise, but recorded as a miss).
- **Findings:**
  - The method scales to a dense ALIF network that is ~15 points more accurate and **fully dependent on recurrence** (12% without it).
  - Certified training moves core-level provable silence from 0% to 58–59%, capturing 93–95% of actual silence, with 0 violations.
  - The trained R_i (0.33–0.40) lands at the Proposition's threshold (1 − β)·θ = 0.3935, as predicted.
- **Next (pre-registered here before running):**
  - **S2-rep:** seeds 2 and 3 at λ ∈ {0, 0.3}, same recipe. Pass: mean cost ≤ 1.0 point and certified ≥ 50% of oracle, 0 violations, across all 3 seeds.
  - **S2b (later):** a stronger recipe aiming for an ≥ 85% control (2 recurrent layers or learnable time constants), to be pre-registered separately.

# OVERNIGHT PLAN (2026-10-06): experiments to close Q1 reviewer gaps. ALL PRE-REGISTERED HERE BEFORE RUNNING.

## ALT-001 — is the certificate loss better than simpler alternatives?
- **Question:** a reviewer will ask "why not just clamp excitatory weights, use L1, or regularize rates harder?"
- **Alternatives** (each trained instead of the certificate loss; everything else identical):
  - **CLAMP:** after every optimizer step, project each neuron's positive recurrent weights so that R_i ≤ 0.33 (the R reached by certified training). This is a hard excitatory-drive budget.
  - **L1:** add μ·mean_i R_i with μ = 0.1.
  - **RATE:** a strong firing-rate penalty, 10·relu(rate − 0.01).
- **Small model** (256 LIF, ring-local, CPU): seeds 1–3, compared against PILOT-005 (λ = 0 and λ = 0.1).
- **Strong model** (512 ALIF dense, GPU): seed 1, CLAMP and L1, compared against S2 (λ = 0 and λ = 0.3).
- **Metrics:** test accuracy, K = 4 certified fraction (neighbour-silent for the small model, core-silent for the strong model), oracle, violations.
- **Pre-stated interpretation rule:**
  - (a) If the certificate loss reaches a higher certified fraction than every alternative at equal or better accuracy (within 1 point), the certificate loss is the method.
  - (b) If CLAMP matches it, the paper presents **excitatory-drive budgeting** (the Proposition's quantity) as the core principle. The certificate loss and the projection become two ways of enforcing it, and the contribution is the certification framework plus the threshold principle.
  - Either outcome is reported as is.

## SSC-001 — second dataset
- Spiking Speech Commands (SSC, 35 classes), same 512-ALIF recipe, 100 bins over 1 s windows. Data is binned lazily per batch from the h5 files. **15 epochs** (compute budget). λ ∈ {0, 0.3}, seed 1.
- **Pass:** the certified model costs ≤ 2 points of accuracy (looser, because of the reduced epochs; stated) and certifies ≥ 50% of the oracle core-silent fraction, with 0 violations.

## FXP-001 — soundness under chip-style fixed-point arithmetic
- Quantize the trained small models (seeds 1–3, λ = 0.1) to integer weights (8-bit, per-layer scale), integer membrane state and integer decay (Loihi-style: v ← v − (v·d >> 12) + input). Run an integer simulation and compute certificates in exact integer arithmetic.
- **Pass:** 0 violations across all certified windows, plus the certified fraction and the accuracy of the quantized network are reported.

## Additional pre-registered experiments (2026-10-06; the user approved unlimited experiments/datasets)

- **LAMBDA-001 (accuracy–certification trade-off curve):**
  - small model, seed 1, λ ∈ {0.01, 0.03, 0.3, 1.0}, plus the existing 0 and 0.1;
  - strong model, seed 1, λ ∈ {0.03, 1.0}, plus the existing 0, 0.1 and 0.3.
  - Reported as a full Pareto curve (certified fraction vs accuracy). No pass/fail; this is descriptive.
- **HORIZON-001 (test of the Proposition):** on saved certified models, compute each neuron's Lemma-1 certified horizon up to 64 steps.
  - Prediction: neurons whose margin m_i = (1 − β)·θ − R_i − max-quiet-input is positive have much longer horizons (median ≥ 4× that of neurons with negative margin).
  - Falsified if the median ratio is < 2×.
- **SCALE-001 (network size):** strong recipe with H ∈ {256, 1024} (512 exists), λ ∈ {0, 0.3}, SHD, seed 1.
  - Pass: certified ≥ 50% of oracle at every size, cost ≤ 1.5 points.
- **ENGINE-002 (speed on the strong dense model):** extend the C++ engine to arbitrary core connectivity (dense: every core waits on all others) for the 512-ALIF S2 models, 16 cores, same latencies.
  - Same pass bar as S1c: handshake/cert ≥ 1.5× at L ≥ 20 µs, and control/certified ≥ 1.3×, exact spikes.
  - ALIF adaptation is simulated exactly, and certificates use the base θ.
- **NMNIST-001 (third dataset, vision; if disk and time allow):** N-MNIST via direct download, a 2-layer version of the strong recipe.
  - Same pass bar as SSC-001.

## HORIZON-001 — RESULT (2026-10-06): PARTIALLY SUPPORTED (ratio 3.4–3.7×; prediction ≥ 4×; falsification < 2×)

| seed | n(m > 0) / n(m ≤ 0) | median horizon m > 0 | median horizon m ≤ 0 | ratio | Spearman(margin, mean horizon) |
|---|---|---|---|---|---|
| 1 | 118 / 138 | 37 | 11 | 3.4× | 0.855 |
| 2 | 116 / 140 | 37 | 11 | 3.4× | 0.831 |
| 3 | 130 / 126 | 37 | 10 | 3.7× | 0.868 |

- The ≥ 4× prediction is **not met**; it is not falsified (> 2×).
- **Strong monotone relation:** margin predicts certified horizon with Spearman 0.83–0.87 on all seeds.
- **Caveat:** horizons are capped by the steps remaining in a 100-step sample (median remaining ≈ 50). The m > 0 group is near that cap, which likely compresses the ratio. This is stated, not used to change the verdict.
- **Paper framing:** "the margin m_i predicts certifiable horizon (ρ ≈ 0.85); positive-margin neurons are certified for a median of 37 steps vs 10–11."

## FXP-001 — RESULT (2026-10-06): **PASS**

- 8-bit integer weights, integer membrane, Loihi-style decay v ← v − ⌊v·1612/4096⌋, θ_int ≈ 317–355.
- **0 violations across 158,571,183 certified neuron-windows** (3 seeds).

| seed | float acc (PILOT-005) | integer acc | neighbour-certified K = 4 | oracle |
|---|---|---|---|---|
| 1 | 69.4 | 68.8 | 58.5% | 62.0% |
| 2 | 67.3 | 67.8 | 57.6% | 61.0% |
| 3 | 68.2 | 68.2 | 57.8% | 61.1% |

- Quantization costs ≤ 0.6 points. Certified fraction ≈ 94–95% of oracle. Certificates are exact in integer arithmetic, as predicted by the monotone-decay argument.

## S2-rep — RESULT (2026-10-06): **PASS** (3 seeds, 512-ALIF dense, SHD, λ = 0.3)

| seed | control acc | certified acc | Δ (pts) | core-cert | oracle | cert/oracle | viol | R | acc, rec. zeroed (cert) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 81.98 | 81.32 | −0.66 | 59.1% | 62.4% | 0.95 | 0 | 0.330 | 11.4% |
| 2 | 80.83 | 76.90 | −3.93 | 60.6% | 63.3% | 0.96 | 0 | 0.333 | 19.1% |
| 3 | 79.37 | 81.58 | +2.21 | 60.9% | 63.1% | 0.97 | 0 | 0.323 | 10.6% |

- **Mean Δ = −0.79 ± 3.08 points** (pass bar: mean cost ≤ 1.0). Certified 95–97% of oracle. 0 violations. **PASS.**
- **Honest note:** seed-to-seed variation (± 3 points) exceeds the mean effect, so the accuracy cost is not distinguishable from zero with 3 seeds. Report the paired mean ± sd and all individual values.
- Controls: core-certified 0.0% on all seeds. Mean control accuracy 80.7% (< 85%, so this remains the "moderate-strength regime").

## S2b — stronger single-layer recipe (PRE-REGISTERED 2026-10-06, before running)

- **Changes from S2:**
  - H = 1024 ALIF;
  - **learnable per-neuron membrane decay β_i** (sigmoid-parameterized, initialized uniformly in [0.5, 0.9]) and learnable adaptation decay ρ_i (heterogeneous time constants; cf. Perez-Nieves et al. 2021);
  - 60 epochs; otherwise the S2 recipe.
- **Certificates:** they use each neuron's own β_i (sound: the Lemma 1 induction is per neuron). The margin becomes (1 − β_i)·θ. The certificate loss uses per-neuron β_i.
- **Arms:** λ ∈ {0, 0.3}, seed 1 (seeds 2–3 if time allows).
- **Pass:** control ≥ 85% (strong regime) **and** certified ≥ 50% of oracle with 0 violations **and** cost ≤ 1.5 points (single seed; seed noise ≈ 3 points is noted).
- **Partial:** the control stays below 85%.

## ALT-001 — INTERIM (small model, seed 1): simple alternatives MATCH the certificate loss (scenario (b))

| method | acc | neighbour-cert K = 4 (recursive) | oracle | R_i mean (median) | acc, recurrence zeroed | viol |
|---|---|---|---|---|---|---|
| control (λ = 0) | 68.2 | 0.0% | 61.1% | 1.445 | 47.2 | 0 |
| certificate loss (λ = 0.1) | 69.4 | 57.9% | 61.3% | 0.327 | 53.4 | 0 |
| **CLAMP** (R_i ≤ 0.33) | 68.5 | **60.1%** | 61.1% | 0.318 (0.328) | 52.8 | 0 |
| **L1** (μ = 0.1) | **70.9** | **59.6%** | 61.2% | 0.043 (0.004) | 53.9 | 0 |

- **Interim reading** (to be confirmed on seeds 2–3 and on the strong model): certifiability comes from **bounding excitatory recurrent drive**, regardless of how it is enforced.
  - L1 drives most excitatory recurrent weights to ~0 (median R_i 0.004), but recurrence stays functional through inhibition (−17 points when zeroed).
- **Per the pre-registered rule (b):** the paper presents the **excitatory-drive budget** (the Proposition's quantity) as the core principle. The certificate loss, a hard projection (CLAMP) and L1 are interchangeable ways to enforce it.
- **Contribution** = certification framework (Lemmas 1–2, adaptive and fixed-point extensions) + the margin/budget principle + exact certificate-based execution with a measured speed-up + the empirical finding that inhibition-dominated recurrence keeps accuracy.

## ALT-001 (strong model) + LAMBDA-001 (strong) — RESULT (2026-10-06, seed 1; control 82.0%, oracle ≈ 62%)

| method | acc (Δ) | core-cert K = 4 | oracle | cert/oracle | R_i mean | firing rate | viol |
|---|---|---|---|---|---|---|---|
| cert λ = 0.03 | 81.54 (−0.4) | 52.1% | 61.5% | 0.85 | 0.572 | 4.43% | 0 |
| cert λ = 0.1 | 80.96 (−1.0) | 58.4% | 62.5% | 0.93 | 0.401 | 4.46% | 0 |
| cert λ = 0.3 | 81.32 (−0.7) | 59.1% | 62.4% | 0.95 | 0.330 | 4.06% | 0 |
| cert λ = 1.0 | 77.52 (−4.5) | 61.9% | 63.5% | 0.97 | 0.264 | 3.66% | 0 |
| **L1** (μ = 0.1) | **81.67 (−0.3)** | 57.6% | 61.9% | 0.93 | 0.142 | 4.19% | 0 |
| **CLAMP** (R ≤ 0.33) | 78.67 (−3.3) | **64.9%** | 67.5% | 0.96 | 0.325 | 1.99% | 0 |

- **Confirms scenario (b) on the strong model:** every method that bounds excitatory recurrent drive yields provable silence (85–97% of oracle, 0 violations).
- The certificate loss gives a smooth accuracy–certification dial. L1 is cheapest in accuracy. CLAMP certifies most, but lowers accuracy and firing rate.
- λ = 0.03 certifies 85% of oracle even with mean R = 0.57 > budget (0.39). K = 4 certification does not need R below the unbounded-horizon budget; the budget governs long horizons (HORIZON-001).
- **Caveat:** single seed; seed noise ≈ ± 3 points, so accuracy rankings between methods are not conclusive.
- **Final framing (pre-registered rule b):** excitatory-drive budgeting is the principle; certificate loss / L1 / projection are enforcement options.

## ENGINE-ALT — speed for alternative enforcement methods (PRE-REGISTERED 2026-10-06, before running)

- The S1c engine (C++, 8 cores, Lemma-1 certificates, exact), run unchanged on the small-model seed-1 **CLAMP** and **L1** models (alt_clamp_s1.pt, alt_l1_s1.pt). Compared against the S1c control and certified numbers.
- **Prediction** (if excitatory-drive budgeting is the principle): handshake/cert ≥ 1.4× at L ≥ 20 µs for both, similar to the certified model (1.50–1.55×).
- **Falsified:** < 1.2× for both at L = 100 µs.
- Run only when no training job is using the CPU (timing hygiene); GPU jobs may run (noted).

## ALT-001 (small model, 3 seeds) — RESULT (rate seed 3 pending)

**DATA-HYGIENE NOTE:** `results/pilot004_l0_s1.json` and `pilot004_l0.1_s1.json` were **overwritten** by PILOT-007 (which used SEED = 1 and the same filename pattern; NP = 16, rate target 1%). The seed-1 control and certified values below are taken from the PILOT-005 logs (task output) recorded in this ledger, not from those files. The seed-2 and seed-3 files are original. Fix: future runs use OUTJSON with unique names.

| method | acc s1 / s2 / s3 (mean) | neighbour-cert K = 4 s1 / s2 / s3 (mean) | firing rate |
|---|---|---|---|
| control | 68.2 / 67.5 / 69.7 (68.5) | 0.0 / 0.0 / 0.0 (0.0%) | 2.4–2.9% |
| certificate loss λ = 0.1 | 69.4 / 67.3 / 68.2 (68.3) | 57.9 / 57.1 / 57.1 (57.4%) | 2.2–2.7% |
| CLAMP R ≤ 0.33 | 68.5 / 68.9 / 66.7 (68.0) | 60.1 / 58.1 / 60.1 (59.4%) | 2.1–2.4% |
| L1 μ = 0.1 | 70.9 / 67.0 / 69.3 (69.1) | 59.6 / 57.6 / 59.1 (58.8%) | 2.3–2.6% |
| **RATE** 10·relu(r − 0.01) | 66.2 / 64.8 / — (65.5) | **0.8 / 0.8 / —** (0.8%) | **1.4–1.8%** |

- **Key negative control:** a strong firing-rate penalty produces the **sparsest** networks (1.4–1.8% firing) but certifies only 0.8%.
- **Sparsity is not certifiability.** Only bounding excitatory recurrent drive makes silence provable.
- All three excitation-bounding methods give 57–59% certified (~95% of oracle), with no meaningful accuracy cost, and 0 violations.

## S2b — RESULT (2026-10-06, seed 1): **PARTIAL** (control 83.9% < 85%); the method holds

| λ | acc | core-cert K = 4 | oracle | cert/oracle | viol | R_i mean | budget mean (1 − β_i)·θ | β_i range | firing rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 83.88 | 0.0% | 56.0% | 0 | 0 | 9.16 | 0.309 | 0.47–0.91 | 4.09% |
| 0.3 | 83.22 (−0.66) | 57.4% | 61.4% | 0.93 | 0 | 0.327 | 0.312 | 0.39–… | 3.01% |

- Heterogeneous learnable time constants and H = 1024 raise the control to 83.9% (+1.9 over S2).
- Certified training costs 0.66 points and certifies 93% of oracle silence, with 0 violations, using per-neuron β_i in the bound.
- Still below the pre-registered 85% "strong" bar, so the strongest single-layer recipe tried remains moderate-strength. SOTA (95–96%) uses delays, multi-layer or SSM architectures that we do not cover; this is stated as a limitation.
- **RATE seed 3:** acc 67.3%, neighbour-cert 0.8% (oracle 62.0%), firing rate **1.32%** (sparsest of all), R_i mean 1.28 > θ, 0 violations.
- **RATE across 3 seeds:** acc 66.2 / 64.8 / 67.3 (mean 66.1), certified 0.8 / 0.8 / 0.8%, firing 1.3–1.8%. **The negative control is confirmed on all seeds.**
- **Timing-hygiene note for ENGINE-ALT / ENGINE-002:** they ran while the SSC GPU job was active (one CPU thread for data binning plus the GPU driver). Possible timing noise is stated in the results.

## ENGINE-ALT — RESULT (2026-10-06): **PASS** (prediction ≥ 1.4× met)

- Small-model seed-1 CLAMP and L1 models; S1c engine; all runs exact. Coverage: CLAMP 67.4%, L1 70.3%.
- Handshake/cert speed-up:
  - CLAMP: 0.65× (L = 0), 1.38× (5 µs), **1.76×** (20), **2.03×** (100), **2.08×** (500);
  - L1: 1.32× (0), 1.39× (5), **1.68×** (20), **2.23×** (100), **2.33×** (500).
- Handshake times are slightly higher than in S1c (e.g. 2.87 vs 2.64 ms at 20 µs), probably because the SSC GPU job used a CPU thread. **Only within-run ratios are compared.** The noise is stated.

## ENGINE-002 — RESULT: **INVALID as a latency test** (design error, stated)

- 16 busy-waiting threads on 12 hardware threads, so the OS scheduling quantum dominated. Per-sample times were 470–575 ms and **flat across L** (latency irrelevant).
- Certificates still gave ~1.5× (468 vs 308 ms; coverage 64.7%), but under artificial over-subscription. **Not claimed.**
- **Corrective rerun (ENGINE-002b), deviation recorded before running:** the same dense 512-ALIF models partitioned as **8 cores × 64 neurons** (fits the hardware threads); same latencies, same exactness check, same pass bar (handshake/cert ≥ 1.5× at L ≥ 20 µs; control/certified ≥ 1.3×).

## ENGINE-002b — RESULT (2026-10-06): **FAIL** (pass bar 1.5×; observed 1.15–1.40×). All runs exact.

| L (µs) | certified: handshake / cert (ms) | speed-up | control: cert (ms) | control/certified (cert mode) |
|---|---|---|---|---|
| 0 | 4.93 / 4.91 | 1.01× | 5.63 | 1.15 |
| 5 | 5.72 / 5.41 | 1.06× | 5.75 | 1.06 |
| 20 | 9.10 / 7.93 | **1.15×** | 8.56 | 1.08 |
| 100 | 18.77 / 13.95 | **1.34×** | 17.57 | 1.26 |
| 500 | 59.10 / 42.22 | **1.40×** | 59.36 | 1.41 |

- Dense 512-ALIF, 8 cores × 64 neurons; per-core certificate coverage 61.4% (control 0%).
- **Interpretation:** with dense inter-core connectivity a core must hear from all 7 other cores, so a step's wait is removed only when every one of them is certified. Benefit is much lower than in ring-local networks (1.5–2.3×).
- **Stated limitation:** the speed-up depends on sparse/local inter-core connectivity (typical of neuromorphic mappings, but not universal).
- Paper claims for speed are restricted to locally connected networks, with this dense result reported alongside.

## SCALE-001 — RESULT (2026-10-06, seed 1, 512-ALIF dense recipe, SHD): **PARTIAL**

| H | control acc | certified (λ = 0.3) acc | Δ | core-cert | oracle | cert/oracle | viol | R_i (ctrl → cert) | acc, rec. zeroed (cert) |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 73.94 | 73.23 | −0.7 | 60.0% | 64.4% | 0.93 | 0 | 3.21 → 0.319 | 46.2% |
| 512 (3 seeds) | 80.7 ± 1.3 | 79.9 ± 2.6 | −0.79 ± 3.07 | 60.2% | 62.9% | 0.96 | 0 | ~4.8 → 0.33 | 10.6–19.1% |
| 1024 | 84.23 | 81.36 | **−2.9** | 60.5% | 63.4% | 0.95 | 0 | 8.87 → 0.374 | 12.0% |

- Certified ≥ 50% of oracle at every size: **met** (93–96%, 0 violations).
- Cost ≤ 1.5 points: met at 256 and 512 (3-seed mean); **missed at 1024** (−2.9, single seed; seed noise ≈ ± 3). Reported as is, with a 1024 replication recommended.
- The non-heterogeneous H = 1024 control reaches 84.2%, slightly above the heterogeneous S2b control (83.9%).

## SSC-001 — λ = 0 control (seed 1, 15 epochs)
- acc 61.6%, core-cert 0.4%, oracle 66.8%, 0 violations, R mean 7.58, firing 1.36%. The λ = 0.3 run is in progress (watcher process).

## SCALE-rep (PRE-REGISTERED 2026-10-06, before running): H = 1024, seeds 2–3, λ ∈ {0, 0.3}
- **Pass:** 3-seed mean cost ≤ 1.5 points, certified ≥ 50% of oracle, 0 violations. Otherwise "cost grows with size" is reported as a finding.

## SSC-001 — RESULT (2026-10-06, seed 1, 15 epochs): **PARTIAL** (cost criterion FAILED)

| arm | acc | core-cert | oracle | cert/oracle | viol | R_i | firing |
|---|---|---|---|---|---|---|---|
| λ = 0 | 61.56% | 0.4% | 66.8% | 0.01 | 0 | 7.58 | 1.36% |
| λ = 0.3 | 57.88% (**−3.7**) | 60.7% | 72.0% | 0.84 | 0 | 0.316 | 0.82% |

- Certification transfers: ≥ 50% of oracle (84%) with 0 violations.
- **Accuracy cost −3.7 > 2.0 bar: FAILED.** Both models are under-trained at 15 epochs; the certified one may converge more slowly. Stated as is.

## SSC-L1 (PRE-REGISTERED 2026-10-06, before running): SSC with L1 on excitatory drive (μ = 0.1)
- Same recipe as SSC-001, 15 epochs, seed 1.
- **Pass:** cost ≤ 2.0 points vs the SSC-001 control (61.56%) and certified ≥ 50% of oracle, 0 violations.

## SCALE-rep — RESULT (2026-10-06): **FAIL** (mean cost 2.11 > 1.5). Reported as a finding: cost grows with size.

| H = 1024 | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|
| control acc | 84.23 | 84.41 | 81.36 | 83.3 ± 1.7 |
| certified (λ = 0.3) acc | 81.36 | 81.14 | 81.18 | 81.2 ± 0.1 |
| Δ (points) | −2.87 | −3.27 | −0.18 | **−2.11 ± 1.68** |
| core-cert / oracle | 60.5 / 63.4 | 59.6 / 62.6 | 62.0 / 64.9 | ≈ 60% (95%), 0 violations |

- **Cost vs size:** H = 256: −0.7 (1 seed); H = 512: −0.79 ± 3.07 (3 seeds); H = 1024: −2.11 ± 1.68 (3 seeds).
- Certified accuracy plateaus near 81% at H = 1024.
- **Hypothesis (not tested):** the per-neuron budget (1 − β)·θ is independent of fan-in, so the constraint tightens relative to capacity as networks grow. Fan-in-aware or layer-wise budgets are future work.

## SSC-L1 — RESULT (2026-10-06): **PARTIAL** (cost bar failed)
- acc 58.52% (−3.0 vs 61.56%), core-cert 56.4%, oracle 67.4% (84%), 0 violations, R 0.038, firing 1.15%.
- On SSC (15 epochs) the cost of bounding excitation is 3–4 points for both the certificate loss and L1.

# OVERNIGHT FINAL STATUS (2026-10-06 07:10)
- All planned experiments are done. The manuscript is updated with every result, including failures (Appendix A).
- **Supported:**
  - the certificate framework and proofs; exactness in float, adaptive and fixed-point arithmetic (0 violations, > 158 M windows);
  - the excitatory-drive budget principle (ALT-001; HORIZON ρ ≈ 0.85);
  - sparsity ≠ certifiability (rate penalty: 0.8%);
  - certification ≈ 60% of core-steps / 84–98% of oracle on every model and dataset;
  - no measurable cost on SHD ≤ 512 neurons;
  - exact speed-up of 1.5–2.3× (local connectivity, L ≥ 20 µs).
- **Not supported / limitations:**
  - cost grows with size (−2.1 at 1024) and on SSC (−3 to −3.7);
  - dense-connectivity speed-up only 1.15–1.40×;
  - emulated latency, no hardware;
  - accuracy below SOTA.

# SWEEP-001 — improving accuracy cost and speed (PRE-REGISTERED 2026-10-06, before running; user-requested 20–30 configs)

**Integrity rules (fixed now):**
1. A **validation split**: 10% of the SHD training set (fixed RNG seed 123), held out from training. All sweep configs are evaluated on validation only. The test set is not used for selection.
2. **Selection rule:** among configs with validation accuracy within 1.0 point of the validation control, choose the highest validation certified core fraction (K = 4). Ties go to the higher validation accuracy.
3. **Confirmation:** the winner (and the λ = 0.3 baseline method for comparison) is retrained on seeds 2–4 and evaluated once on the test set. Only the confirmation numbers are headline numbers.
4. **All sweep configs are reported** in a supplementary table.

**Base model:** 512 ALIF, dense, 16 cores × 32, SHD, 40 epochs, seed 1, as in S2.

**A — sink hubs ("activity-aware partitioning"):**
- The first n_hub cores are hub cores: excluded from the excitation penalty, they **receive from all cores but do not project to other cores** (recurrent weights from hub neurons into non-hub neurons are masked to 0); they still feed the readout.
- Non-hub cores are constrained.
- Grid: n_hub ∈ {1, 2, 4} × λ_cert ∈ {0.3, 1.0} × enforcement ∈ {certificate loss, L1 μ = 0.1 (λ ignored)}. L1 at μ = 0.1 and μ = 0.3 is used as the "λ" pair, giving **12 configs**.

**C — recursive-aware (activity-weighted) loss:**
- R_i^eff = α·R_i(all) + (1 − α)·Σ_j relu(W_ij)·a_j, where a_j = 1 if neuron j spiked in the window (detached).
- Certification at evaluation always uses the sound Lemma 1/2 bounds.
- Grid: α ∈ {0, 0.5} × λ ∈ {0.1, 0.3, 1.0}, giving **6 configs**.

**Reference configs:** control; certificate loss λ = 0.3; L1 μ = 0.1; CLAMP (validation versions), giving **4 configs**.
**Total: 22 training configs.**

**B — distributed recursive certificates** (engine; no training): a core's certificate uses its neighbours' published certificates to drop their excitation from the bound for the steps they are certified silent (sound: certificates are facts about future silence).
- Evaluated on the existing small models (certificate loss, CLAMP, L1) and on the dense 512 model, and later on the SWEEP winner.
- Cores wait only on cores that actually project to them (needed for sink hubs).
- Pass bar for B: ≥ 1.2× additional speed-up over the S1c/ENGINE-002b certificates at L = 100 µs for the dense model.

## B (distributed recursive certificates) — RESULT (2026-10-06): **FAILED** pre-registered bar (dcert/cert at L = 100 µs, dense = 0.73× vs ≥ 1.2×)

- All runs exact.
- **Coverage gain is negligible:**
  - small_cert 67.4 → 68.1%; small_clamp 67.4 → 68.9%; small_l1 70.3 → 70.6%;
  - dense_cert 61.4 → 63.1%; dense_ctrl 0 → 0%.
- **dcert is slower than plain cert for every model** (dcert/cert 0.32–0.86×). Per-step recomputation, which loops over all presynaptic cores, costs more than the tiny coverage gain saves.
- **Why:** a core's horizon is limited by its own neurons' worst-case drive and by external input, not by neighbours' excitation.
- **Timing hygiene:** run concurrently with two GPU sweep jobs (CPU contention). Absolute times are inflated and noisy (e.g. small_cert handshake 4.75 ms vs 2.64 ms in S1c), and plain-cert speed-ups vary across runs (small_cert at L = 20: 1.10× here vs 1.50× in S1c). Only within-run comparisons are used. The final speed numbers for the sweep winner will be measured on an idle CPU.
- **Conclusion:** the simple per-core Lemma-1 certificate is the right run-time choice. The distributed variant is reported as a negative result.

## SWEEP-001 — selection (validation, 20/22 configs done; the last two are C variants)
- **Winner by the pre-registered rule: `ref_clamp` (excitation cap R_i ≤ 0.33, no hubs):** val acc 94.36 (control 93.50), val certified 66.52% (oracle 69.69%), 0 violations.
- Runner-up: A_h4_cert03 (sink hubs + certificate loss): 93.25 / 65.83%.
- The original method ref_cert03 (92.02 / 61.41%) is **not eligible** (−1.5 pts on validation).
- **User decision (2026-10-06):** adopt the winner as "our architecture", named **ExCap** (Excitation-Capped recurrent SNN), conditional on confirmation.

## CONFIRM-001 (PRE-REGISTERED, before running)
- **Arms:**
  - ExCap (ref_clamp) — winner;
  - certificate loss λ = 0.3 (original method);
  - control;
  - A_h4_cert03 (sink hubs) — **labelled exploratory**.
- Seeds 2, 3, 4. TEST = 1 (test set evaluated once per run). Same 512-ALIF recipe; the validation split is still held out.
- **ExCap is confirmed if**, over 3 seeds, the mean test accuracy cost vs control is ≤ 1.0 point **and** test certified core fraction ≥ 60%, with 0 violations.
- Otherwise the paper reports ExCap as the validation winner that did not confirm.
- **SWEEP-001 FINAL (22/22):** the selection rule picks **ref_clamp = ExCap** (val 94.36%, certified 66.52%). The last two C configs (C_a05_l01, C_a0_l10) did not change the outcome. Full table: `results/sweep_table.csv`. CONFIRM-001 launched (two queues).

## FXP-ExCap — RESULT (2026-10-06): **PASS**
- ExCap small models (alt_clamp_s1–3), 8-bit integer weights, Loihi-style integer decay.
- **0 violations / 159,158,750 certified neuron-windows.**
- Integer acc 68.8 / 68.7 / 67.0% (float 68.5 / 68.9 / 66.7).
- Neighbour-certified 60.8 / 58.7 / 60.8% vs oracle 61.8 / 60.3 / 61.5%, i.e. **97–99% of oracle**.

## CONFIRM-001 — INTERIM (seeds 2–3 of 2–4): **ExCap FAILS confirmation**
- Test acc, seeds 2 / 3:
  - control 81.27 / 80.34;
  - **ExCap 76.06 / 76.50 (mean −4.5)**;
  - cert λ = 0.3: 78.53 / 78.67 (−2.2);
  - hub4: 78.49 / 78.93 (−2.1).
- Test certified: ExCap 65.0 / 65.8%; cert 60.6 / 60.9%; hub4 63.6 / 62.6%. 0 violations everywhere.
- ExCap cannot meet the ≤ 1.0 mean-cost bar even if seed 4 costs 0.
- **Diagnosis:** the validation split (random 10% of training) contains the *same speakers* as training, while ~80% of the SHD test set comes from 2 unseen speakers (4, 5). ExCap's hard cap appears to hurt speaker generalization, which same-speaker validation cannot detect. **Methodological fix below.**

# SWEEP-002 — improving accuracy (PRE-REGISTERED 2026-10-06, before running)

**Selection fix:** a **speaker-disjoint validation set**: training speakers {3, 6} held out (1,169 samples). This mimics the unseen-speaker test condition.

**Techniques:**
- **FT:** fine-tune from the trained control instead of training from scratch. 20 epochs, lr 5e-4, constraint weight ramped linearly over the first 10 epochs.
- **KD:** knowledge distillation from the trained control (weight 1, temperature 2) added to the loss.

**Configs (seed 1, 512-ALIF dense, SHD):**
- S0 control (scratch, 40 ep; also the teacher and init)
- S1 certificate loss λ = 0.3, scratch
- S2 FT + cert λ = 0.3
- S3 FT + cert λ = 0.3 + KD
- S4 FT + cert λ = 1.0 + KD
- S5 scratch + cert λ = 0.3 + KD
- S6 FT + 4 sink hubs + cert λ = 0.3 + KD
- S7 FT + ExCap + KD

**Selection rule:** the highest speaker-val certified fraction among configs within 1.0 point of S0's speaker-val accuracy. If none qualifies, take the config with the smallest accuracy cost among those certifying ≥ 55%.

**Confirmation:**
- seeds 2–4: control → winner (using that seed's control as init/teacher), plus scratch cert λ = 0.3; test set.
- Pass: mean test cost ≤ 1.0 point, certified ≥ 55%, 0 violations.

## MACHINE MIGRATION — RTX 2060 -> RTX 5080 (logged 2026-10-06, before rerunning SWEEP-002)

**New environment (frozen for all runs from here on):**
- GPU: RTX 5080 16 GB (Blackwell, sm_120 native in torch's arch list), driver 616.56; CPU 32 threads; 23 GB RAM.
- OS/toolchain: WSL2 Ubuntu 26.04, Python 3.14, **torch 2.14.1+cu130**, g++ 15.2 (C++20).
- Previous environment: RTX 2060 6 GB + i7-10750H (6 cores), older CUDA build.
- Deviations from `setup_5080.sh`: `sudo` is password-protected on this machine, so pip was bootstrapped
  without it (`python3 -m venv --without-pip` + `get-pip.py`); `g++` was already installed. The script's
  cu128 pin has no cp314 wheel for Python 3.14, so **cu130** was used instead.
- `~/research/venv_speech` is a symlink to `venv_gpu` (disk: the WSL vhdx sits on E:, which has ~8.5 GB free).
  Verified safe: every script on the venv_speech path is device-agnostic or explicitly `map_location="cpu"`.
- Environment verified: sm_120 is natively supported (not JIT), GPU matmul matches CPU to 3.2e-5.
- Throughput: **5 s/epoch** for the 512-ALIF SHD run (40 epochs ~ 4 min).

**Why SWEEP-002 stage 1 is rerun here rather than reusing the 2060's S0:**
the 2060's S0 control finished (speaker-val 73.05%) but its weights `sw2_S0_ctrl_s1.pt` were never
committed (models are gitignored), and stage 2 needs them as `INIT`/`TEACHER`. S0 must therefore be
retrained on this machine; S1 is rerun too so the whole sweep is internally consistent in one environment.
The 2060 result is archived as `results/sw2_S0_ctrl_s1_rtx2060.json` and used **only** as a cross-machine
sanity check, never mixed into the sweep.

**Pre-registered cross-machine check (stated before running):** the 5080 S0 control should reach a
speaker-disjoint validation accuracy **within 3.0 points of 73.05%**. A larger divergence means the new
environment changed training behaviour and must be diagnosed before any sweep result is trusted.
Seeded runs are *not* expected to match bit-exactly across GPU architecture and torch version.

**Why the protocol survives the migration:** every reported accuracy cost is computed against a control
trained in the *same* environment, so cross-machine drift in absolute accuracy does not bias the costs.

**Engineering change, no scientific effect:** `run_sw2_stage2.sh` goes from 2 queues of 3 to 6 parallel
queues of 1 (16 GB allows it). Identical seeds, configs and selection rule.

**WARNING for the speed results:** the manuscript's 1.5-2.3x engine speed-ups were measured on the
6-core i7-10750H. This machine has 32 threads and `-march=native` targets a different CPU, so CPU
wall-clock numbers from the two machines **must not be mixed in one table**. Re-measuring the whole
speed section here, or keeping the i7 numbers, is an open decision.

**Noted weakness in the SWEEP-002 selection rule (rule unchanged, not revised post-hoc):** the
speaker-disjoint validation set has 1,169 samples, so the standard error of its accuracy is ~0.88 points
-- comparable to the rule's 1.0-point eligibility band. Which config wins may therefore be partly noise;
CONFIRM-002 on fresh seeds against the test set is the real guard. Also note the speaker-disjoint
validation set is *harder* than the test set (control 73.05% val vs 80-81% test), so it is a conservative
but not perfectly calibrated proxy.

## SWEEP-002 — RESULT (2026-10-06, RTX 5080, seed 1, speaker-disjoint validation only)

Cross-machine check **passed**: S0 control 74.68% vs the RTX 2060's 73.05% (+1.63, inside the
pre-registered 3.0-point band). 0 violations in every config. Budget (1-beta)*theta = 0.3935.

| tag | val acc | cost | certified | oracle | R_mean |
|---|---|---|---|---|---|
| S0_ctrl              | 74.68 | +0.00 |  0.0% | 63.5 | 4.776 |
| S1_cert03 (scratch)  | 71.34 | -3.34 | 63.5% | 66.4 | 0.331 |
| **S2_ft_cert03**     | **73.99** | **-0.68** | **58.0%** | 65.2 | 0.388 |
| S3_ft_cert03_kd      | 74.76 | +0.09 | 53.7% | 64.6 | 0.461 |
| S4_ft_cert10_kd      | 73.22 | -1.45 | 59.4% | 65.3 | 0.345 |
| S5_scratch_cert03_kd | 72.54 | -2.14 | 60.9% | 65.4 | 0.401 |
| S6_ft_hub4_kd        | 73.40 | -1.28 | 56.0% | 65.8 | 1.602 |
| S7_ft_excap_kd       | 72.63 | -2.05 | 64.9% | 66.1 | 0.329 |

**Winner by the pre-registered rule: S2_ft_cert03** (eligible within 1.0 point; highest certified
fraction among eligible). Rule application verified by hand: eligible set = {S2 (-0.68), S3 (+0.09)};
S2 certifies 58.0% vs S3's 53.7%.

**Main finding — initialization, not the loss, caused most of the accuracy cost.** The identical
constraint at the identical strength costs 3.34 points from scratch (S1) but 0.68 points as
fine-tuning from the trained control (S2), while still certifying 58.0% (89% of oracle). This is the
first result that brings the accuracy cost under the 1-point bar on the harder unseen-speaker condition.

**Knowledge distillation is counterproductive for certifiability.** S3 (= S2 + KD) attains the best
accuracy of all (+0.09 above control) but its R_mean rises to 0.461, *above* the 0.393 budget, and
certification falls to 53.7%. KD pulls the network back toward the unconstrained teacher and so fights
the constraint. Mechanistically consistent with Proposition 1 (certifiability is governed by R_i vs the
budget), and a useful negative result: accuracy recovery and certifiability are not automatically aligned.

**ExCap remains the certifiability champion and remains too expensive.** S7 certifies 64.9% (98% of its
oracle) at -2.05 points, still ineligible -- consistent with its CONFIRM-001 failure on unseen speakers.

**Selection fragility (flagged BEFORE running, now quantified).** With a ~0.88-point standard error on
1,169 validation samples, S2 clears the 73.68 eligibility boundary by only 0.31 points and S4 misses it
by 0.46 points while certifying *better* (59.4%). The top-3 ordering is therefore inside noise, and the
winner must be treated as provisional until CONFIRM-002 on fresh seeds against the test set.

### Process issue found and fixed (2026-10-06) — silent failure masking in the harness
The first attempt at SWEEP-002 appeared to succeed (exit 0, no errors) but produced **no checkpoints**,
so all six stage-2 configs -- which fine-tune from S0 -- had no `INIT`/`TEACHER` and died instantly,
also silently. Cause: every run was piped through `grep -E "^(RESULT|epoch ...)"`, which discarded
tracebacks, so a failing `torch.save` was invisible. Because the result JSON is written *immediately
before* the checkpoint, the failed run still left a normal-looking result file.
**Fix:** `run_sw2_stage1.sh`, `run_sw2_stage2.sh` and `run_confirm2.sh` now write full logs to
`~/research/logs/<tag>.log`, print the RESULT lines after the fact, and exit non-zero with a tail of the
log if a result line or an expected checkpoint is missing. Stage 2 aborts if the teacher checkpoint is
absent; each CONFIRM-002 chain aborts if the control a later run depends on failed.
The exact mechanism by which the pipe killed the save (most likely SIGPIPE propagation on consumer
close) was **not** proven; the fix is verified to work, and a harness that discards tracebacks is worth
removing regardless. **Other scripts still using the `| grep` form should be migrated before reuse.**

## CONFIRM-002 — RESULT (2026-10-06, RTX 5080, seeds 2-4, test set, one-shot): **PASS**

Protocol as pre-registered: recipe selected on speaker-disjoint validation (seed 1), confirmed on fresh
seeds 2-4 against the test set. Winner = **S2_ft_cert03** (fine-tune from that seed's own control,
20 epochs, lr 5e-4, ramped constraint, cert lambda = 0.3).

| arm (mean of seeds 2-4) | test acc | cost | certified | % of oracle | R_mean (budget 0.3935) |
|---|---|---|---|---|---|
| control                      | 78.77 |  --   |  ~0.00% |  --   | 4.80 |
| **ours: FT + cert 0.3**      | 78.34 | **-0.43** | **55.98%** | 89.3% | 0.390 |
| reference: scratch + cert 0.3| 77.68 | -1.09 | 60.45% | 95.2% | 0.330 |

Per-seed test accuracy -- control / ours / reference:
seed 2: 78.53 / **79.42** / 78.31; seed 3: 78.58 / 77.16 / 78.22; seed 4: 79.20 / 78.45 / 76.50.
Per-seed certified (ours): 55.01 / 56.14 / 56.79. **0 violations in every arm and seed.**

**Pre-registered bar (mean cost <= 1.0 point, certified >= 55%, 0 violations): met on all three.**
Cost -0.43 (on seed 2 the constrained model beat its control by +0.88), certified 55.98% with every
seed individually above 55%, zero violations. Controls certify ~0%.

**Mechanism confirms Proposition 1 rather than merely correlating with it:** mean R_mean settles at
0.3904 against a budget of 0.3935, i.e. training drives worst-case excitatory drive to just under the
threshold, which is exactly the condition the proposition says makes silence provable.

**Main conclusion -- the accuracy cost was caused by initialization, not by the loss.** Identical loss
at identical strength costs -1.09 points from scratch and -0.43 as fine-tuning. This resolves the
project's main open problem (ExCap's -4.5 failure in CONFIRM-001).

**Secondary result -- an accuracy/certifiability trade-off knob.** Scratch training certifies more
(60.45%, 95% of oracle) at roughly double the accuracy cost; fine-tuning certifies less (55.98%, 89% of
oracle) at half the cost. Both are defensible operating points and should be reported as a pair rather
than one winner.

**Honest limits of this result:**
1. With 3 seeds the per-seed costs span +0.88 to -1.41, so the -0.43 mean is **not statistically
   distinguishable from zero, nor from a cost of a point or two**. The defensible claim is "no
   measurable accuracy cost at this size on SHD", not "no cost". More seeds are cheap (~4 min each).
2. One dataset, one size (512 ALIF, SHD). The documented cost growth at 1,024 neurons (-2.1) and on
   SSC (-3 to -3.7) was measured with **from-scratch** training and has **not** been retested with
   fine-tuning. Whether the fix generalizes is now the highest-value open question.

### Methodological finding — the speaker-disjoint protocol costs ~2 points of absolute accuracy
Holding out speakers 3 and 6 removes 1,169 training samples (14.3%) and 2 of the 10 training speakers.
Measured effect on the control's test accuracy: **80.8% (CONFIRM-001, random val split, 10 speakers)
vs 78.8% (CONFIRM-002, speaker-disjoint, 8 speakers)**, i.e. about -2.1 points.
Consequence for the manuscript: the speaker-disjoint split is the correct instrument for *selecting* a
recipe (it is what exposed ExCap's failure), but it handicaps the absolute numbers. **The final reported
model should be retrained on the full training set using the confirmed recipe**, with the
speaker-disjoint runs reported as the selection/confirmation protocol. Reporting 78.3% as the method's
accuracy would understate it by ~2 points for a reason unrelated to the method.
Accuracy *costs* are unaffected, since every cost is computed against a control trained identically.

# SCALE-FT-001 — does the fine-tuning fix survive at 1,024 neurons? (PRE-REGISTERED 2026-10-06, before running)

**Why:** CONFIRM-002 removed the accuracy cost at 512 ALIF on SHD by applying the constraint as
fine-tuning instead of from scratch. The manuscript's most damaging limitation is that the cost *grows*
with size (-2.1 points at 1,024, 3 seeds) -- but that was measured with **from-scratch** training. If
fine-tuning also removes it at 1,024, the strongest reviewer objection ("the principle degrades exactly
where it would matter") largely dissolves. If it does not, the limitation stands and must be reported.

**No selection occurs in this experiment.** The recipe is already fixed by CONFIRM-002, so this is a
confirmation run evaluated once on the test set against a bar stated here in advance.

**Configuration:** H = 1,024 (32 cores x 32 neurons), SHD, speaker-disjoint val (speakers 3, 6 held out),
seeds 1-3, TEST = 1. Arms per seed:
- control: scratch, 40 epochs;
- **ours:** fine-tune from that seed's control, 20 epochs, lr 5e-4, constraint ramped over 10 epochs,
  cert lambda = 0.3 (the CONFIRM-002 recipe, unchanged);
- reference: scratch + cert lambda = 0.3, 40 epochs (the from-scratch arm that previously cost -2.1).

**Pass (stated before running):** mean test accuracy cost of *ours* vs control <= 1.0 point, mean test
certified core fraction >= 55%, 0 violations.

**Pre-registered interpretations:**
- **Pass:** the fix is size-robust; the "cost grows with size" limitation is retired for SHD and the
  manuscript's limitation section and Table of scale results must be rewritten.
- **Fail with ours clearly better than the reference (-2.1):** fine-tuning helps but does not fully fix
  the size trend; report the partial improvement honestly and treat fan-in-aware budgets as the next
  method step.
- **Fail with ours ~= the reference:** fine-tuning does not generalize beyond 512; the limitation stands
  as currently written and the CONFIRM-002 result must be scoped explicitly to 512.

# CONFIRM-002b — seed extension (PRE-REGISTERED 2026-10-06, before running)

**Why:** CONFIRM-002 passed on seeds 2-4, but the per-seed accuracy costs span +0.88 to -1.41, so the
-0.43 mean is not statistically distinguishable from zero or from a 1-2 point cost. Three more seeds
roughly halves the standard error.

**Anti-bias commitments, stated before running (adding seeds after seeing a result is how optional
stopping creeps in):**
1. Seeds **5, 6, 7** are run, and **all** of them are reported, whatever they show.
2. **No stopping rule based on the outcome.** Seeds are not added or dropped after inspecting results.
3. The headline figure becomes the mean over **all six seeds (2-7)**; the seeds 2-4 mean stays on record
   in this file so the change is auditable.
4. Identical recipe, protocol and arms to CONFIRM-002 -- only the seed differs. No reselection.
5. If the six-seed mean cost exceeds the original 1.0-point bar, **CONFIRM-002 is downgraded to a
   partial pass** and the manuscript must say so.

**Arms per seed:** control (scratch, 40 ep); ours (FT from that seed's control, 20 ep, lr 5e-4, ramp,
cert 0.3); reference (scratch + cert 0.3, 40 ep). H = 512, SHD, speaker-disjoint val, TEST = 1.

**GPU note:** run concurrently with SCALE-FT-001. Concurrency affects wall-clock only -- seeds, batch
size (128), precision (fp32) and all hyperparameters are unchanged, so results are unaffected. Batch
size and precision were deliberately **not** raised to increase GPU usage, because both would alter the
optimization trajectory/numerics and break comparability with CONFIRM-002 and all earlier results.

# SSC-FT-001 — does the fine-tuning fix survive on the second dataset? (PRE-REGISTERED 2026-10-06, before running)

**Why:** the manuscript reports a -3 to -3.7 point cost on SSC, its worst accuracy result, measured with
**from-scratch** training at **15 epochs**. CONFIRM-002 showed that on SHD the cost was caused by
from-scratch initialization, not by the loss. SSC is therefore the second half of the generalization
question, and the existing SSC number confounds two things: initialization *and* a short training budget.

**Confound to separate:** this experiment matches the SHD protocol (control 40 epochs, ours = FT 20
epochs, reference = scratch 40 epochs), so it tests fine-tuning *and* retires the separate "SSC with
40+ epochs" open item at the same time. The 15-epoch scratch result stays on record for comparison.

**Configuration:** DATASET = ssc (35 classes), H = 512, seeds 1-3, TEST = 1. Data staged at
`/mnt/d/research_data/ssc` (symlinked from `~/research/data/ssc`) because the WSL disk had only ~8.4 GB
free and this project has a logged disk-full corruption incident. Arms per seed:
- control: scratch, 40 epochs;
- **ours:** FT from that seed's control, 20 epochs, lr 5e-4, ramp, cert lambda = 0.3 (CONFIRM-002 recipe,
  unchanged -- no retuning for this dataset);
- reference: scratch + cert lambda = 0.3, 40 epochs.

**Note:** SSC has no speaker metadata equivalent to the SHD split used for selection. **No selection is
performed here** -- the recipe is fixed by CONFIRM-002, so this is a confirmation run against a bar set
in advance.

**Pass (stated before running):** mean test accuracy cost of *ours* vs control <= 1.5 points (a looser
bar than SHD's 1.0, stated in advance because SSC is a 35-class task with a weaker control), mean test
certified core fraction >= 50%, 0 violations.

**Pre-registered interpretations:**
- **Pass:** the fix is dataset-robust; the manuscript's worst accuracy limitation is retired.
- **Fail but clearly better than -3 to -3.7:** fine-tuning helps on SSC without eliminating the cost;
  report the partial gain and keep the limitation in weakened form.
- **Fail at ~= the scratch cost:** the fix is SHD-specific; CONFIRM-002 must be explicitly scoped to SHD
  and the generalization claim dropped.

**Step 0 (gate before the full run):** a 1-epoch timing probe. SSC is ~9x SHD's sample count and is read
lazily per batch from an h5 on drvfs (`/mnt/d`), whose random-read performance is unverified. If a probe
epoch exceeds ~3 minutes, the h5 must be moved onto the ext4 disk (after compacting the vhdx to reclaim
space) before committing to 9 runs, rather than burning hours on I/O stalls.

### Infrastructure incident (2026-10-06) — CUDA driver faults at 6 concurrent processes in WSL
**What happened:** GPU concurrency was raised from 3 to 6 CUDA processes to increase GPU usage. Five runs
then died with `torch.AcceleratorError: CUDA error: unknown error`
(`CUDA_ERROR_UNKNOWN` / 999 from `cuMemcpyHtoDAsync_v2`): all three SCALE-FT-001 fine-tuned arms at
H = 1024 and two of the three CONFIRM-002b controls. **Not** out of memory -- no log contains an OOM
message and peak usage was 9.9 of 16.3 GB. Consistent with WSL2 GPU paravirtualization becoming unstable
under many concurrent CUDA contexts.

**Throughput measurement that makes this purely a loss:** at H = 512, 1 job = 5-6 s/epoch, 3 jobs = 8
s/epoch (~1.9x net throughput), 6 mixed jobs = 19-20 s/epoch, i.e. **total throughput flat-to-worse**
while GPU power fell from 149 W to 136 W of a ~360 W budget. This workload is launch-latency bound (100
sequential timesteps over small tensors), so extra processes divide capacity instead of adding to it.

**Standing constraint adopted: at most 3 concurrent CUDA processes on this machine.** It is both faster
per job and stable. `run_rerun_failed.sh` encodes this.

**Integrity assessment of the surviving runs:** the failures were hard process crashes, not silent
corruption, and CUDA contexts are isolated between processes, so cross-contamination is implausible. The
surviving arms completed normally, wrote both result and checkpoint, and their values are in the expected
ranges (H = 1024 control 79.4-80.4%, scratch-cert 76.4-78.1%, R_mean 0.386-0.401, 0 violations). They are
treated as valid. Note these runs are not bitwise reproducible across reruns anyway (cuBLAS/cuDNN
nondeterminism is not disabled), so an exact-match recheck is not available.

**Deliberately NOT done to raise GPU usage:** batch size and precision (bf16 / torch.compile) were left
unchanged, because both alter the optimization trajectory or numerics and would break comparability with
CONFIRM-002 and all earlier results; precision additionally affects the R_i bound arithmetic that the
soundness guarantee depends on.

### SCALE-FT-001 — partial result (surviving arms, seeds 1-3, H = 1024, test set)
| arm | test acc | cost | certified | % of oracle | R_mean |
|---|---|---|---|---|---|
| control        | 79.77 |  --   |  0.00% |  --   | 8.72 |
| ref (scratch)  | 77.00 | -2.77 | 59.99% | 95.2% | 0.392 |
| **ours (FT)**  | *lost to the CUDA incident; rerunning* | | | | |

Confirms the manuscript's "cost grows with size" for **from-scratch** training (-2.77 here vs -2.1
reported). The fine-tuned arm -- the actual question -- is being rerun.

**Mechanistic finding (new): the size effect is a fan-in effect.** Control R_mean rises from 4.78 at
H = 512 to **8.72** at H = 1024, i.e. **1.83x for 2x width**, while the budget (1-beta)*theta = 0.3935 is
fixed. Worst-case excitatory drive therefore grows roughly linearly with fan-in and the constraint becomes
proportionally tighter as the network widens. This explains *why* the accuracy cost grows with size and
directly motivates **fan-in-aware budgets** (scaling the budget with fan-in, or constraining per-fan-in)
as the principled next method step rather than an arbitrary extension.

## SCALE-FT-001 — RESULT (2026-10-06, H = 1024, SHD, seeds 1-3, test set): **FAIL**

| arm | test acc | cost | certified | % of oracle | R_mean (budget 0.3935) |
|---|---|---|---|---|---|
| control        | 79.77 |  --   |  0.00% |  --   | 8.72 |
| **ours (FT)**  | 79.30 | **-0.47** | **0.40%** | 0.6% | **0.510** |
| ref (scratch)  | 77.00 | -2.77 | 59.99% | 95.2% | 0.392 |

Per-seed cost / certified for ours: -0.71 / 0.74%, -0.35 / 0.45%, -0.35 / 0.01%. 0 violations everywhere.

**Pre-registered bar (cost <= 1.0, certified >= 55%, 0 violations): FAILS on certification.**
Cost passes comfortably (-0.47); certification is 0.40% against a 55% bar.

**Interpretation: fine-tuning preserved accuracy by not actually satisfying the constraint.** R_mean
finished at 0.510, *above* the 0.3935 budget. Certification is a threshold phenomenon in R vs the budget
(Proposition 1), so ending 30% above the budget yields ~0% certified rather than a proportionally reduced
figure. This is Proposition 1 behaving exactly as stated -- the mechanism is confirmed even though the
experiment failed.

**Consistent with the fan-in finding.** The H = 1024 control sits at R_mean 8.72 vs 4.78 at H = 512, while
the budget is fixed. A 20-epoch fine-tune at lr 5e-4 cannot cover the larger distance to the same budget,
so it stalls above it.

**Consequences:**
1. **CONFIRM-002's result must be explicitly scoped to H = 512 on SHD.** The generalization claim is not
   supported.
2. **The manuscript's "accuracy cost grows with size" limitation STANDS** and must remain in the paper.
   At H = 1024 the only certifiable configuration measured is from-scratch training at -2.77 points.
3. The trade-off at 1024 is currently binary: accuracy (FT, no certificates) **or** certificates
   (scratch, -2.77 points). There is no known setting that gives both at this width.

**Shortcoming in this experiment's own pre-registration (recorded, not retrofitted):** the three stated
interpretations were all framed in terms of *accuracy cost* and none anticipated the observed mode --
accuracy preserved, certification lost. Future pre-registrations for constrained training must state
pass/fail jointly over **both** axes and name the "constraint not actually satisfied" outcome explicitly.

# SCALE-FT-002 — is the 1024 failure merely an insufficient fine-tuning budget? (PRE-REGISTERED 2026-10-06, before running)

**Hypothesis:** SCALE-FT-001's fine-tuned arm stalled at R_mean 0.510 vs the 0.3935 budget -- only ~30%
above it. A longer fine-tune or a stronger constraint weight may push R under the budget while keeping
most of the accuracy advantage over from-scratch training (-0.47 vs -2.77).

**Protocol discipline: selection on the speaker-disjoint VALIDATION set only.** SCALE-FT-001 already spent
a test evaluation at this width. Tuning the fine-tuning budget against the test set would be selection on
test, which this project forbids. Therefore:
- **Stage 1 (validation only, seed 1, H = 1024):** fine-tune from the existing seed-1 control over
  EPOCHS x CERT_LAMBDA in {20, 40} x {0.3, 1.0} (4 configs; the {20, 0.3} cell is SCALE-FT-001's setting
  and is re-run for a like-for-like validation comparison).
- **Selection rule:** among configs reaching **R_mean <= 0.3935** *and* validation certified >= 50%, take
  the highest validation accuracy. If none reaches the budget, the experiment **fails** and no test
  evaluation is performed.
- **Stage 2 (confirmation, only if stage 1 yields a winner):** seeds 2-3 on the test set, one shot.
  **Pass:** mean test cost <= 1.5 points (looser than 512's 1.0, stated in advance because the scratch
  alternative at this width costs -2.77) AND mean test certified >= 55% AND 0 violations.

**Pre-registered interpretations:**
- **Pass:** the 1024 failure was a budget artefact; the size limitation weakens to "needs a longer
  fine-tune at larger widths", and the recipe becomes width-dependent rather than broken.
- **Stage 1 fails to reach the budget:** fine-tuning cannot satisfy the constraint at this width at any
  tested budget. The size limitation stands as written, and **fan-in-aware budgets** (scaling the budget
  with fan-in, as the R ~ fan-in finding implies) become the required method change rather than an
  optional extension.
- **Stage 2 fails:** report that R can be brought under budget at 1024 but only at an accuracy cost
  comparable to from-scratch training, i.e. fine-tuning's advantage is 512-specific.

## CONFIRM-002b — RESULT (2026-10-06, seeds 5-7 added; H = 512, SHD, test set): **PASS, but the headline cost roughly doubles**

All six seeds reported as pre-registered (no stopping rule, no reselection).

| seed | control | ours (FT+cert 0.3) | cost | certified | oracle | R_mean | viol |
|---|---|---|---|---|---|---|---|
| 2 | 78.53 | 79.42 | **+0.88** | 55.01% | 62.21 | 0.389 | 0 |
| 3 | 78.58 | 77.16 | -1.41 | 56.14% | 62.38 | 0.409 | 0 |
| 4 | 79.20 | 78.45 | -0.75 | 56.79% | 63.49 | 0.373 | 0 |
| 5 | 79.99 | 79.11 | -0.88 | 54.99% | 61.81 | 0.407 | 0 |
| 6 | 77.47 | 75.75 | -1.72 | 53.94% | 62.40 | 0.388 | 0 |
| 7 | 79.77 | 78.18 | -1.59 | 57.77% | 62.79 | 0.369 | 0 |

**Six-seed mean cost -0.91** (sd 0.96, se 0.39, 95% CI **[-1.68, -0.14]**); mean certified **55.77%**
(range 53.94-57.77); **0 violations in all six seeds**.

**Pre-registered bar (mean cost <= 1.0, certified >= 55%, 0 violations): PASS on all three.**

**But the interpretation changes, and the earlier claim must be withdrawn:**
1. The seeds 2-4 mean was **-0.43**; all three added seeds were worse (-0.88, -1.72, -1.59), moving the
   mean to **-0.91**. The standard error halves (0.68 -> 0.39) exactly as intended.
2. **The 95% CI now excludes zero.** The claim "no measurable accuracy cost" (recorded in the 12:45
   brief status on the 3-seed data) is **no longer supportable and is withdrawn.** The defensible claim is
   **"a cost of about 0.9 points (95% CI 0.1-1.7)"**.
3. The mean passes the 1.0-point bar, but **the bar lies inside the confidence interval**, so a true cost
   above 1.0 cannot be excluded. Report the bar as met by the point estimate, not as a demonstration that
   the cost is below 1.0.
4. One seed (6) certifies 53.94%, below 55%; the *mean* of 55.77% is what the pre-registration specified,
   so the criterion is met, but the per-seed spread should be reported.

**Why this matters methodologically:** the 3-seed estimate was optimistic by ~0.5 points, i.e. by more
than the entire remaining margin to the bar. This is a concrete demonstration of why the project's
anti-optional-stopping rule exists -- had the seeds been added and then selected on, the favourable
3-seed figure would have survived into the manuscript.

**Fine-tuning's advantage over from-scratch training is still real but smaller than first measured:**
-0.91 (FT, 6 seeds) vs -1.09 (scratch, 3 seeds). The two are now close, and a 6-seed scratch arm would be
needed before claiming FT is meaningfully better on accuracy at H = 512. The certifiability gap remains
clear (55.8% FT vs 60.5% scratch).

## SCALE-FT-002 stage 1 — RESULT (2026-10-06, H = 1024, seed 1, validation only): the 1024 failure WAS a budget artefact

Control validation accuracy at H = 1024: 76.13 (mean, seeds 1-3). Budget 0.3935. All figures below are on
the **speaker-disjoint validation set**, so they are comparable with each other (the -2.77 figure quoted
earlier for scratch is a *test* number and must not be compared with these directly).

| config | val cost | certified | R_mean |
|---|---|---|---|
| scratch lam=0.3 (n=3)        | -5.59 | 63.55% | 0.392 |
| FT lam=0.3, 20 ep (n=3)      | -3.99 |  0.59% | 0.510 |
| **FT lam=0.3, 40 ep (n=1)**  | **-2.48** | **54.03%** | 0.395 |
| FT lam=1.0, 20 ep (n=1)      | -4.88 | 59.95% | 0.345 |
| FT lam=1.0, 40 ep (n=1)      | -5.82 | 62.07% | 0.309 |

**SCALE-FT-001's conclusion is REVISED.** "Fine-tuning does not generalize to 1,024 neurons" was an
artefact of the **20-epoch** fine-tuning budget, not a property of fine-tuning. At 40 epochs with the same
lambda = 0.3, fine-tuning at H = 1024 costs **-2.48** and certifies **54.03%**, against scratch's -5.59 at
63.55%: **less than half the accuracy cost at a comparable certifiable operating point.** A longer
fine-tune is what the larger fan-in distance required, exactly as the R ~ fan-in analysis implied.

**Flaw in this experiment's own pre-registered selection rule (disclosed, not worked around).** The rule
gated eligibility on `R_mean <= 0.3935`. That excluded FT lam=0.3/40 ep (R_mean = 0.395) by 0.4% -- yet
that config certifies 54%. **R_mean is the wrong gate:** certification is evaluated per neuron, so a mean
marginally above the budget still leaves many neurons below it. The rule's literal winner is
FT lam=1.0/20 ep (-4.88, 59.95%), which is *dominated on accuracy* by the config the rule excluded.
The better config is **not** silently substituted; both are carried into a replication with a corrected
rule (below).

# SCALE-FT-003 — replication of the two 1024 candidates with a corrected rule (PRE-REGISTERED 2026-10-06, before running)

**Why:** the stage-1 candidates are single-seed (n = 1), and the stage-1 eligibility gate was
mis-specified (see above). Both candidates are therefore replicated before any test evaluation.

**Corrected eligibility rule:** gate on the quantity actually of interest -- **validation certified core
fraction >= 50%** -- rather than on `R_mean <= budget`. R_mean is retained as a *reported diagnostic*,
not as a gate. Rationale: certification is per-neuron; a mean marginally above the budget can still
certify a large fraction, which stage 1 demonstrated empirically.

**Runs:** validation only, H = 1024, seeds 2-3 (seed 1 already done in stage 1), for both candidates:
- FT lambda = 0.3, 40 epochs, lr 5e-4, ramp, init from that seed's control;
- FT lambda = 1.0, 20 epochs, lr 5e-4, ramp, init from that seed's control.

**Selection (3 seeds each, validation):** among configs with mean validation certified >= 50%, take the
**smallest mean accuracy cost**. Report both configs' full numbers regardless of which wins.

**Stage 2 (one shot, test set, seeds 1-3):** the selected config, plus the scratch lambda = 0.3 arm already
measured at this width, for a like-for-like comparison.
**Pass:** mean test cost **<= 3.0 points** AND mean test certified **>= 50%** AND 0 violations. The 3.0-point
bar is set in advance and is deliberately looser than the 512 bar of 1.0, because the alternative
certifiable configuration at this width (scratch) costs -5.59 on validation; the question here is whether
fine-tuning *materially reduces* that cost, not whether it eliminates it.

**Pre-registered interpretations:**
- **Pass:** the size limitation weakens from "the fix fails at scale" to "the fix needs a longer
  fine-tune at larger fan-in, and costs ~2-3 points at 1,024 instead of ~5.6". The manuscript's
  limitation section must be rewritten accordingly.
- **Fail:** fine-tuning's advantage does not survive replication at this width; SCALE-FT-001's original
  negative conclusion stands and must be reported as such.

## SCALE-FT-003 — RESULT (2026-10-06, H = 1024, validation, 3 seeds each)

**CORRECTION to the stage-1 entry above.** The "-2.48, less than half the cost" figure recorded for
FT lambda=0.3/40 ep (seed 1) was computed against the **mean** control validation accuracy (76.13)
instead of **seed 1's own** control (77.41). Seed-matched, that run costs **-3.76**, not -2.48. All
figures below are seed-matched. The stage-1 conclusion that "the 1024 failure was a budget artefact" is
**withdrawn**: a longer fine-tune helps, but far less than that error suggested.

| config (validation, seed-matched, n=3) | mean cost | mean certified | mean R_mean |
|---|---|---|---|
| **FT lambda=0.3, 40 ep** | **-4.02** | **55.82%** | 0.399 |
| FT lambda=1.0, 20 ep | -6.10 | 61.37% | 0.347 |
| scratch lambda=0.3 | -5.59 | 63.55% | 0.392 |
| FT lambda=0.3, 20 ep (SCALE-FT-001) | -3.99 | 0.59% | 0.510 |

Per-seed costs / certified -- FT lam0.3/40ep: -3.76/54.03%, -3.59/56.58%, -4.70/56.86%.
FT lam1.0/20ep: -6.16/59.95%, -5.13/62.64%, -7.01/61.53%. **0 violations in all 12 runs.**

**Corrected pre-registered rule (certified >= 50%, smallest cost) selects FT lambda=0.3, 40 epochs.**

**Honest conclusion: SCALE-FT-001's negative finding largely STANDS.** At H = 1024, fine-tuning for 40
epochs buys only **+1.57 accuracy points** over from-scratch training (-4.02 vs -5.59) while certifying
**7.7 points less** (55.82% vs 63.55%). That is a mild trade-off along the same curve, not a fix. The
dramatic initialization advantage measured at H = 512 (-0.91 FT vs -1.09 scratch, with the from-scratch
*validation* cost at -3.34) **does not carry to 1024**.

**Note on the 20-epoch result:** FT lambda=0.3/20 ep has essentially the same *validation accuracy cost*
as 40 ep (-3.99 vs -4.02) but certifies 0.59% instead of 55.82%. The extra 20 epochs buy certification at
no additional accuracy cost -- the constraint needs the time to pull R under the budget, and until it does,
the accuracy has already been paid without the benefit being obtained. **This is the clearest evidence yet
that R vs budget is a threshold, not a gradient (Proposition 1).**

**Process note:** this is the second baseline error of the session in the same direction (over-optimistic).
Both were caught by seed-matching and replication. Any future cost figure must be computed per seed against
that seed's own control, never against a pooled control mean.

# CORE-SCALING-001 — exact speed-up vs core count (PRE-REGISTERED 2026-10-06, before running)

**Why:** the manuscript's headline speed figures (1.5-2.3x local, 1.15-1.40x dense) were measured at
**8 cores on a 6-core i7-10750H** -- oversubscribed, and the plan already records one invalid engine run
from exactly that cause. This machine has **32 threads**, so core counts 2..32 can be measured without
oversubscription for the first time. The paper's motivation is that *synchronization cost grows with
system size*; if the certificate advantage does **not** grow with core count, the motivation undercuts
the result, and if it does, that rising curve is the single most valuable figure for a
parallel-computing venue.

**Setup:** CONFIRM-002's confirmed-recipe model (FT + cert lambda 0.3, H = 512, seed 2) and its control.
`CORES` in {2, 4, 8, 16, 32} (all divide 512). The engine sweeps interconnect latency
L in {0, 5, 20, 100, 500} us internally and compares mode 0 (local handshake) against mode 1 (certificate),
verifying spike trains **bit-for-bit** against a single-thread reference on every run.

**Timing hygiene (mandatory):** the script blocks until no GPU training process remains, because CPU
wall-clock measurement requires an idle machine. All earlier engine timings taken alongside GPU jobs are
treated as within-run comparisons only.

**Pre-registered predictions and interpretations:**
- **Speed-up rises with core count** (expected): supports the paper's core motivation and becomes the
  headline scaling figure. Report the curve at each latency.
- **Speed-up flat in core count:** the "synchronization cost grows with scale" motivation is not
  demonstrated by our own engine; the claim must be weakened to a fixed-size statement and the venue
  strategy reconsidered.
- **Speed-up falls with core count:** report as a negative result; the mechanism does not scale and the
  paper must say so explicitly.
- **Any bit-mismatch vs the reference is a soundness failure** and overrides all speed results.

## FAILURE ANALYSIS — where does the accuracy actually go? (2026-10-06, `analyze_accuracy_loss.py`, `analyze_ei_ablation.py`)

**The mechanism, measured.** The trained constraint is R_i = sum_j relu(W_ij) <= (1-beta)*theta = 0.3935.
That budget is a **fixed absolute quantity** -- it does not grow with the network -- while the required
drive does: R_mean = 4.66 at H = 512 and 8.72 at H = 1024. Compliance therefore requires a
**12x (H=512) to 22x (H=1024) reduction in total positive recurrent weight mass**.

Measured on c2_ctrl_s2 vs c2_ours_s2 (H = 512, 300-400 test samples, CPU):

| quantity | control | constrained |
|---|---|---|
| positive recurrent weight mass | 2385.5 | **199.1** (12.0x smaller) |
| negative recurrent weight mass | 3672.2 | **4448.0** (grew) |
| E/I ratio | 0.650 | **0.045** |
| R_mean (all-fire bound) | 4.659 | 0.389 |
| neurons under budget | 0.0% | 64.1% |

**So the constraint works by near-eliminating recurrent excitation while inhibition grows.**

**What recurrence actually contributes (E/I ablation, same models):**

| variant | control acc | constrained acc |
|---|---|---|
| full recurrence | 80.00 | 80.00 |
| excitation removed (negatives kept) | 62.25 (**-17.75**) | 69.25 (**-10.75**) |
| inhibition removed (positives kept) | 5.00 (-75.00), firing rate explodes to 71% | 17.50 (-62.50) |
| no recurrence | 11.75 (-68.25) | 14.75 (-65.25) |

Recurrent **inhibition is essential** (removing it causes runaway firing and collapse to chance).
Recurrent **excitation is worth ~17.8 points** in the control. The constrained model depends on excitation
**less** (-10.75 vs -17.75), i.e. fine-tuning lets the network **re-route around** the excitation it loses.
**This is why fine-tuning beats from-scratch training: it preserves a working solution and adapts it,
rather than having to learn one under the handicap.**

**Hypothesis tested and REJECTED: tightening the bound via an activity cap is not sufficient.**
The all-fire bound is enormously pessimistic -- measured simultaneous activity is **0.98 of 32 neurons per
core on average, p99.9 = 7, max = 10**, against a bound assuming all 512 fire. A sound "at most k per
presynaptic core" bound (sum of the k largest positive weights per core) was computed directly:

| k per core | control R_k | tightening | neurons under budget |
|---|---|---|---|
| 1 | 1.125 | 4.1x | 0.0% |
| 2 | 1.809 | 2.6x | 0.0% |
| 4 | 2.760 | 1.7x | 0.0% |
| 10 (observed max) | 4.201 | 1.1x | 0.0% |

Even the most aggressive possible cap (k = 1, one spike per core per step) leaves R = 1.125, still
**2.9x above the budget**, and certifies 0% of neurons without retraining. The tightening is sublinear
because the positive weight distribution is heavy-tailed -- the largest few weights carry most of the mass.
**Activity-cap / k-WTA certificates are therefore NOT the fix.** (Recorded so this is not re-attempted.)

### Solutions implied by the analysis, in order of measured leverage

1. **More, smaller cores (no retraining required).** CORE-SCALING-001 shows certificate coverage rising
   **0.4% -> 1.4% -> 16.4% -> 40.3% -> 55.8%** as cores go 2 -> 4 -> 8 -> 16 -> 32 at fixed H = 512.
   Certification requires a **whole core** to be provably silent, so fine-grained cores qualify far more
   often. This is the largest lever found and it costs no accuracy at all -- it is a partitioning choice.
   **Design implication: the technique favours fine-grained many-core architectures.**
2. **Faster membrane leak.** The budget is (1-beta)*theta with beta = exp(-0.5) = 0.6065 today.
   Halving the membrane time constant gives beta = exp(-1) and a budget of 0.632 (**1.61x**); quartering it
   gives beta = exp(-2) and 0.865 (**2.20x**). A single hyperparameter, fully sound, no new theory. Cost:
   shorter membrane memory. **Untested.**
   (Note: raising theta alone does *not* help -- scaling theta and all weights together leaves R/budget
   unchanged. The leak is not scale-invariant, which is why it works.)
3. **Local / sparse recurrent connectivity.** R sums positive weights over *all* presynaptic cores;
   restricting each neuron to a few neighbouring cores cuts R roughly in proportion (~8x at 16 cores).
   This is also already the best speed-up regime (1.5-2.3x local vs 1.15-1.40x dense) and PILOT-004
   reached 56% certification at -2.2 points with local connectivity. **Untested at this scale.**
4. **Longer fine-tune at larger width** -- now demonstrated (SCALE-FT-003 stage 2).

Combining (2) and (3) would raise the budget ~2.2x while cutting required R ~8x, i.e. roughly a **17x**
relaxation against the 12-22x shortfall -- plausibly certification at near-zero accuracy cost, without
crushing excitation. **This is the highest-value method direction and it follows from measurement rather
than preference.**

## CORE-SCALING-001 — RESULT (2026-10-06, H = 512 seed 2, idle CPU, 32 threads): the advantage GROWS with core count

Speed-up = local-handshake time / certificate time. All runs verified **bit-identical** to a single-thread
reference (exact=1 throughout).

| model | cores | L=0 | L=5us | L=20us | L=100us | L=500us | cert coverage |
|---|---|---|---|---|---|---|---|
| ours | 2  | 0.99 | 0.97 | 0.99 | 1.00 | 1.00 | 0.4% |
| ours | 4  | 0.98 | 0.96 | 0.99 | 1.00 | 1.00 | 1.4% |
| ours | 8  | 0.97 | 1.02 | 1.04 | 1.02 | 1.00 | 16.4% |
| ours | 16 | 1.01 | 1.05 | 1.04 | 1.01 | 1.00 | 40.3% |
| **ours** | **32** | 19.66* | 1.24 | **1.67** | **1.71** | 1.41 | **55.8%** |
| control | 32 | 1.25 | 1.24 | 1.03 | 0.99 | 0.95 | **0.0%** |

*The L = 0 value of 19.66x is treated as suspect (measurement artefact at zero emulated latency) and is
not used in any claim.

**This supports the paper's central motivation using our own engine:** synchronization cost grows with
system size, and so does the certificate advantage. The control gains nothing at any core count (0%
coverage), so the entire effect is attributable to training. The gain at 32 cores (1.41-1.71x, dense
connectivity) also **exceeds the manuscript's dense figure of 1.15-1.40x**, which was measured at 8 cores
on a 6-core CPU (oversubscribed).

**Mechanism: certification is governed by core granularity.** A whole core must be provably silent, so
cores of 16 neurons (CORES = 32) qualify far more often than cores of 256 (CORES = 2).
**Honest caveat:** real neuromorphic cores hold far more neurons (Loihi ~1k), where whole-core silence
would be rarer. The regime this technique wins in is **many fine-grained cores**, and the paper must say so.

## SCALE-FT-003 stage 2 — RESULT (2026-10-06, H = 1024, TEST set, seeds 1-3): PASS

| arm | test cost | certified | oracle | violations |
|---|---|---|---|---|
| **FT lambda=0.3, 40 ep** | **-0.44** | **53.13%** | 61.70 | 0 |
| scratch lambda=0.3 | -2.77 | 59.99% | 62.99 | 0 |

Per-seed cost: -1.63, -0.22, **+0.53**. Pre-registered bar (cost <= 3.0, cert >= 50%, 0 violations):
**met on all three.** Fine-tuning gains **+2.33 accuracy points** over from-scratch at this width, for
6.9 points less certification.

**The H = 1024 limitation is therefore substantially addressed by a longer fine-tune**, reversing
SCALE-FT-001's negative conclusion (which used 20 epochs).

**Caveat that must be reported: the cost is speaker-dependent.** The same configuration costs **-4.02 on
speaker-disjoint validation** (held-out speakers 3, 6) but **-0.44 on test** (81% speakers 4, 5). A
3.6-point gap, well beyond the ~0.9-point validation standard error. The precise accuracy cost therefore
depends on which speakers are held out, and both figures must appear in the manuscript rather than only
the favourable one.
