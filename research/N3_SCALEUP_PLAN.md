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
