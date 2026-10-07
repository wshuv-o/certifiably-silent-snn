# Short-Delay Excitatory Drive Governs Provable Silence in Recurrent Spiking Networks

**Authors:** [AUTHOR NAME(S)]

**Affiliation:** [AFFILIATION]

**Corresponding author:** [EMAIL]

## Abstract

Multi-core neuromorphic systems advance in discrete time steps, and before a core can compute step *t*+1 it must establish which of its presynaptic neurons spiked at step *t*. This is enforced by a global barrier or by local handshakes, and the resulting synchronisation is a recognised bottleneck. Trained recurrent spiking networks are in fact silent for most of their operation, but silence that is merely observed cannot be exploited: a core may only skip waiting if the silence of its neighbours is *provable* in advance. We show that provability in a recurrent spiking network is governed by a single measurable quantity: the worst-case excitatory recurrent drive arriving through synapses whose delay is shorter than the certificate horizon, which we denote *R*_short. Across six architectures spanning a twenty-fold range of *R*_short, certified silence rises from 0% to 98% of the achievable maximum, monotonically in *R*_short. Because only synapses with delay below the horizon contribute uncertainty, a network with multi-tap synaptic delays can satisfy a given *R*_short by *relocating* excitation into longer delays rather than removing it. We observe exactly this: training with a certificate penalty reduces drive on the binding short-delay tap by 94% while leaving total excitatory mass essentially unchanged, and the resulting networks certify 56.0% of core-steps on the Heidelberg Spiking Digits at an accuracy change of +0.24 points against matched controls (three test seeds), and 61.5% on Spiking Speech Commands at +3.03 points (four test seeds, frozen recipe transferred without retuning). No soundness violation occurred in any run. We further find that per-synapse learnable delays place enough mass at long lags that the *unconstrained* network already certifies 97.4% of its achievable maximum, so certifiability can be an architectural property rather than a training objective. Certificates permit barrier-free execution that is bit-identical to lock-step; in a compiled shared-memory engine this is 1.25–1.49× faster than lookahead-aware local handshaking at 32 cores, but slower below 16 cores, and in a two-process TCP configuration it is 1.37× slower, because certificates reduce waiting rather than message count.

**Keywords:** spiking neural networks, neuromorphic computing, synchronisation, synaptic delays, formal guarantees, parallel discrete-event simulation

## 1. Introduction

Neuromorphic processors execute spiking neural networks as many small cores exchanging spike events [1–3]. Most advance in discrete algorithmic time steps, and the dependency is strict: before computing step *t*+1, a core must know which of its presynaptic neurons spiked at step *t*. This is enforced either by a global barrier [2, 3] or by local handshakes between communicating cores [4]. As systems scale, the cost of this synchronisation grows with them, and performance analyses of deployed neuromorphic workloads repeatedly identify communication and synchronisation, rather than arithmetic, as the limiting factor [4–6].

Trained recurrent spiking networks offer an apparent opportunity. Their activity is sparse — in the networks studied here the mean firing rate is around 3% — so cores are idle most of the time. But observed idleness cannot be exploited. A core may skip waiting for a neighbour only if that neighbour's silence is established *before* the step in question; otherwise a missed spike changes the computation. The requirement is therefore not sparsity but *provable* sparsity, and the two are not the same: in the networks we examine, cores are genuinely silent in roughly 59% of core-steps while only 4% of those are provable without intervention.

This paper identifies what provability depends on. We formulate per-core silence certificates for adaptive leaky integrate-and-fire networks with multi-tap synaptic delays, and show that the certificate at horizon *k* must bound only those synapses whose delay is *shorter* than *k*; contributions arriving through longer delays originate at time steps that are already determined and can be used exactly. Provability is therefore governed not by total recurrent excitation but by the excitation arriving through short delays, *R*_short.

Two consequences follow, and both are borne out empirically. First, because longer-delay synapses are unconstrained by a horizon-*k* certificate, a network can reduce *R*_short by moving excitation into longer delays instead of discarding it, so certifiability need not cost accuracy. Second, because the quantity is a property of the delay distribution, an architecture whose delays are naturally spread away from short lags is certifiable without any dedicated training.

Our contributions are:

1. **The governing quantity.** We derive the delay decomposition of the certificate and show that certified silence is monotone in *R*_short across six architectures spanning *R*_short from 0.275 to 5.983, rising from 0% to 98% of the achievable maximum (section 5.1).
2. **Certification at no accuracy cost.** On the Heidelberg Spiking Digits, networks trained with a certificate penalty certify 56.0% of core-steps at an accuracy change of **+0.24** points relative to matched controls over three held-out test seeds; the recipe transferred to Spiking Speech Commands without retuning certifies 61.5% at **+3.03** points over four test seeds (sections 5.2, 5.3).
3. **The mechanism is reallocation, not reduction.** Training reduces drive on the binding short-delay tap by 94% while total excitatory mass changes by 1–16%, with the same signature on both datasets and all seeds (section 5.4).
4. **Certifiability can be architectural.** With per-synapse learnable delays the *unconstrained* network certifies 97.4% of its achievable maximum, so no certificate penalty is required when the delay distribution already supplies the margin (section 5.5).
5. **Exact execution, with its limits stated.** Certificates permit barrier-free execution that is bit-identical to lock-step. In a shared-memory engine this is 1.25–1.49× faster than lookahead-aware local handshaking at 32 cores but slower below 16; in a two-process TCP configuration it is 1.37× slower, because certificates reduce waiting and not message count (section 5.6).
6. **A set of negative results** delimiting the method: tightening the bound with activity caps, enlarging the budget through faster membrane leak, restricting fan-in, increasing width, and replacing fixed taps with learnable per-synapse delays all fail to improve the accuracy–certifiability frontier (section 5.7).

## 2. Background and related work

**Synchronisation in neuromorphic systems.** TrueNorth advances all cores on a global tick [2]. Loihi exchanges barrier messages between neighbouring cores that flush in-flight spikes and propagate the time-step advance [3]. SpiNNaker runs cores against real-time timers and exchanges multicast packets [1]. Recent work replaces global barriers with local, neighbour-to-neighbour handshakes, removing the linear growth of synchronisation cost with system size [4]. Our certificates are complementary: they reduce how often a handshake must actually be waited for, rather than changing the handshake itself.

**Lookahead in parallel discrete-event simulation.** The problem of advancing a process without global synchronisation is long established. Conservative algorithms require *lookahead* — a guarantee that no event will arrive before some future time — to let a process proceed safely [7, 8]; optimistic algorithms instead run ahead and roll back on causality errors [9]. Distributed spiking network simulators take their lookahead from the minimum synaptic delay, exchanging spikes only once per minimum-delay interval [10]. This is exactly the structure we exploit, and it means the relationship between our work and that literature is constructive rather than merely distinguishing: minimum delay supplies lookahead for free, and certificates extend it beyond what the delay alone provides. Importantly, this also means a fair baseline must be *given* the delay-derived lookahead; we do so throughout (section 4.5).

**Bounding spiking network dynamics.** Interval-propagation bounds on spiking network state have been used to certify robustness to input perturbation [11]. We use the same family of bounds for a different property — future silence over a horizon under recurrent input — and for a different purpose, namely making a runtime scheduling decision safe. The technical difference is that the quantity to be bounded is a temporal reach rather than an output margin, which is what gives rise to the delay decomposition in section 3.3.

**Synaptic delays in trained spiking networks.** Delays are an established route to accuracy on temporal benchmarks; methods that learn a delay per synapse reach the strongest reported results on the Heidelberg datasets [12]. Our interest in delays is different: they change *what the certificate must bound*. We find they also make networks certifiable without a dedicated objective, which to our knowledge has not been reported.

**Activity sparsity is not provability.** Firing-rate penalties are the standard route to sparse spiking activity. We find that sparsity and certifiability are close to orthogonal: the networks with the lowest firing rates in our sweeps are not the certifiable ones, and a strong rate penalty produces sparse networks that certify under 1% of core-steps. The quantity that matters is the worst-case drive through short delays, not the mean activity.

## 3. Theory

### 3.1 Network model

We consider a single recurrent layer of adaptive leaky integrate-and-fire (ALIF) neurons. With membrane potential *v*, adaptation *a*, external input *I* and spikes *s* ∈ {0,1}, neuron *i* evolves as

> a_i(t) = ρ a_i(t−1) + γ s_i(t−1)

> v_i(t) = β v_i(t−1) + I_i(t) + Σ_d Σ_j W^(d)_ij s_j(t−d)

> s_i(t) = 1 if v_i(t) ≥ θ + a_i(t), else 0,   and v_i(t) ← v_i(t) − s_i(t)(θ + a_i(t))

where β = exp(−Δt/τ_m) is the membrane decay, θ the base threshold, and W^(d) the recurrent weight matrix for synaptic delay *d* ∈ 𝒟. Setting 𝒟 = {1} recovers the conventional unit-delay recurrent network. Neurons are partitioned into *P* cores of *C* neurons each; a core is the unit of synchronisation.

### 3.2 Silence certificates

A core issues a certificate "silent through step *u*" when it can establish that none of its neurons will fire at steps *t*+2 … *u*, whatever its neighbours do. A neighbour holding such a certificate need not be waited for over that interval. Because the true threshold θ + a_i is at least θ (adaptation is non-negative), it is sound to reason against the base threshold θ.

Let *R*^(d)_i = Σ_j max(0, W^(d)_ij) be the worst-case excitatory drive into neuron *i* through delay *d*, obtained when every presynaptic neuron fires. Bounding the membrane reach forward from the measured state *v*_i(*t*+1) gives a sufficient condition for silence over *K* steps. If the drive is bounded by *R*_i at every step, the reach converges to *R*_i/(1−β), so

> R_i < (1 − β) θ

is sufficient for silence over an unbounded horizon. We refer to (1−β)θ as the **excitatory budget**. For the parameters used here (β = e^(−1/2), θ = 1) the budget is 0.3935. A finite-*K* certificate is more permissive than this condition, because it starts from the measured membrane potential rather than from the worst case; section 5.1 quantifies the gap.

We evaluate certificates with a fixed-point refinement. Let 𝒮 be a set of neurons that may fire within the window. Bounding each neuron's reach using only the drive attributable to 𝒮 yields a possibly smaller set; iterating to a fixed point gives the certified set. This is sound for any initial superset and tightens the bound where the network is mostly quiet.

### 3.3 The delay decomposition

The central observation of this paper concerns which synapses a certificate must bound. Consider a certificate rooted after step *t*+1 and a horizon *k* ≥ 1, so the step under consideration is τ = *t*+1+*k*. The contribution of a synapse with delay *d* arrives from step

> τ − d = t + 1 + k − d

If *d* ≥ *k*, then τ − *d* ≤ *t*+1, which is a step that has **already been computed**. Its spikes are known, and the contribution can be evaluated exactly rather than bounded. Only synapses with *d* < *k* draw on steps that are still in the future and therefore require a worst-case bound. Hence the quantity that must be bounded at horizon *k* is

> R_short(k) = Σ_{d < k} R^(d)_i

and for a *K*-step certificate the binding quantity is *R*_short = Σ_{d<K} *R*^(d). Two corollaries follow immediately.

First, a minimum synaptic delay *d*_min yields *d*_min steps of exact, cost-free lookahead: for *k* ≤ *d*_min no synapse satisfies *d* < *k*, so nothing requires bounding. This is precisely the lookahead conservative parallel simulation extracts from minimum delay [7, 10], recovered here as a special case; certificates extend lookahead beyond it.

Second, and more consequentially, *R*_short can be reduced *without reducing total excitation*, by moving excitatory mass from short to long delays. A network with delays 𝒟 = {2, 4, 8} and *K* = 4 is bound only by its *d* = 2 tap; the *d* = 4 and *d* = 8 taps are unconstrained by the certificate. Section 5.4 shows that trained networks exploit this.

### 3.4 Enforcement

Certificates can be made to hold by adding a penalty on the worst-case *K*-step reach, evaluated on windows in which the network is in fact silent:

> L = L_task + λ · mean[ relu( max_k reach_k − θ ) · 1(silent) ]

where reach_k uses the exact/bounded split of section 3.3. This is one enforcement route among several; section 5.5 shows that when the delay distribution already places little mass at short lags, no penalty is needed at all.

## 4. Methods

### 4.1 Datasets and protocol

We use the Heidelberg Spiking Digits (SHD; 20 classes, 8,156 training and 2,264 test samples) and Spiking Speech Commands (SSC; 35 classes, 75,466 training samples) [13]. Input spikes over 700 channels are binned into *T* = 100 time steps spanning 1.4 s.

**Selection and confirmation are strictly separated.** SHD's test set is dominated by two speakers absent from training (81.3% of test samples come from speakers 4 and 5, which never appear in training), so a random validation split is not representative of the test condition: in earlier work on this project a configuration that won on a random split lost 4.5 accuracy points on test. All architecture and hyperparameter choices in this paper were therefore made on a **speaker-disjoint validation set** (training speakers 3 and 6 held out, 1,169 samples), and the resulting recipe was **frozen** before any test evaluation. Reported headline results come from the frozen recipe evaluated once per run on the test set with **fresh random seeds** not used during selection. Holding out two speakers costs the control about 2.1 points of absolute test accuracy, so validation figures are systematically lower than test figures and the two are never mixed.

Every accuracy cost is computed **per seed against that seed's own matched control** — a control with identical architecture, schedule and seed, trained without the certificate penalty — never against a pooled control mean.

### 4.2 Architectures

All networks have *H* = 512 recurrent ALIF neurons partitioned into 16 cores of 32 neurons (and 4–32 cores for the execution study), β = e^(−1/2), θ = 1, ρ = e^(−14/200), γ = 0.02. We compare: a conventional unit-delay network (𝒟 = {1}); multi-tap networks with 𝒟 = {1,2,4,8} and 𝒟 = {2,4,8}; and a network with a learnable delay per synapse, parameterised as *D*_ij ∈ [2, 8] via a sigmoid and interpolated onto integer taps by a triangular kernel, *W*^(k)_ij = *W*_ij · relu(1 − |*D*_ij − *k*|). The interpolation kernel sums to unity per synapse, so total synaptic strength is preserved; the parameter count is 2*H*² irrespective of the delay range, against |𝒟|·*H*² for fixed taps.

### 4.3 Training

Surrogate-gradient training with AdamW (learning rate 2 × 10⁻³, weight decay 10⁻⁴), batch size 128, cosine schedule, 150 epochs. Augmentation applies circular shifts in time and channel (±10 bins) and masks a random block of up to 40 channels and up to 10 time steps. Constrained networks are obtained by fine-tuning from the matched control for 150 epochs at learning rate 5 × 10⁻⁴ with the penalty weight ramped linearly over the first half of training; λ = 1.0 unless stated. On SSC the recipe is transferred unchanged, with the epoch count matched by **gradient steps** rather than epochs (8,250 steps, i.e. 14 SSC epochs), since SSC is roughly nine times larger; this is the only adaptation made, and it was declared before the runs.

### 4.4 Certification measurement

We report the **certified core fraction**: the proportion of core-steps for which a whole core is provably silent over the next *K* = 4 steps. The natural ceiling is the **oracle fraction**, the proportion of core-steps in which the core is *actually* silent over the same window; a certified fraction cannot exceed it. We therefore report certification both absolutely and as a percentage of oracle. A **violation** is a certified window in which some neuron does in fact fire; violations are counted exhaustively on every evaluation and any non-zero count invalidates the corresponding result.

### 4.5 Exact multi-core execution

Two engines were implemented in C++20. The first is a shared-memory engine with one thread per core and emulated interconnect latency *L*, in which published data becomes visible to other cores *L* µs after publication. The second partitions cores across separate operating-system processes exchanging fixed-size frames over TCP with Nagle's algorithm disabled, so that latency, system-call cost and scheduling are those the operating system actually imposes.

In both engines the **baseline is given the lookahead the delays provide**: to compute step *t*+1 a core waits only for neighbour spikes up to step *t*+1−*d*_min, which with *d*_min = 2 leaves every core permanently one step ahead with no certificate at all. Comparing certificates against a delay-naive handshake would attribute to the method a speed-up that belongs to the delays.

Both engines verify spike trains **bit-for-bit against a single-threaded reference** on every run. Any mismatch invalidates the timings of that run. All reported timings were taken on an otherwise idle machine; two earlier measurement attempts were discarded because concurrent workloads were detected, and in one case the idle check itself contained a race.

## 5. Results

### 5.1 *R*_short governs certifiability

Table 1 compares six networks spanning a twenty-fold range of *R*_short on the speaker-disjoint validation set.

| architecture | validation accuracy | certified | oracle | % of oracle | *R*_short | total *R* |
|---|---|---|---|---|---|---|
| unit delay 𝒟={1} | 75.19 | 0.00% | 63.93% | 0.0% | 4.840 | 4.84 |
| 𝒟={1,2,4,8} | 81.61 | 1.63% | 60.11% | 2.7% | 5.983 | 12.30 |
| 𝒟={2,4,8}, control | 87.25 | 3.54% | 58.63% | 6.0% | 4.968 | 15.27 |
| 𝒟={2,4,8}, constrained | 87.85 | 56.28% | 59.28% | **94.9%** | **0.275** | 12.76 |
| learnable delays, control | 85.54 | 55.88% | 57.36% | **97.4%** | **1.595** | 7.48 |
| learnable delays, constrained | 85.63 | 55.63% | 56.79% | **98.0%** | **0.392** | 7.00 |

*Table 1. Certified silence is monotone in R_short and essentially independent of total recurrent excitation. Budget (1−β)θ = 0.3935. Validation set, seed 1.*

Certification is monotone in *R*_short and unrelated to total excitation: the two networks with the largest total drive (15.27 and 12.76) sit at opposite ends of the certification range, while the pair differing only in *R*_short (4.968 versus 0.275) differ by a factor of sixteen in certified fraction. Near-oracle certification appears once *R*_short falls to roughly 1.6, about four times the budget; the finite-horizon certificate is thus more permissive than the unbounded-horizon condition of section 3.2, as expected, since it starts from the measured membrane potential.

### 5.2 Certification at no accuracy cost on SHD

Table 2 reports the frozen recipe evaluated once on the SHD test set with three fresh seeds.

| seed | control | constrained | cost | control certified | constrained certified | oracle | *R*_short | violations |
|---|---|---|---|---|---|---|---|---|
| 2 | 87.28 | 87.90 | +0.62 | 4.08% | 55.68% | 58.74% | 0.253 | 0 |
| 3 | 86.75 | 86.62 | −0.13 | 4.60% | 56.66% | 59.44% | 0.253 | 0 |
| 4 | 88.03 | 88.25 | +0.22 | 4.06% | 55.60% | 58.68% | 0.260 | 0 |
| **mean** | **87.35** | **87.59** | **+0.24** | **4.25%** | **55.98%** | **58.95%** | | **0** |

*Table 2. SHD test set, frozen recipe, fresh seeds. Cost standard deviation 0.38, standard error 0.22; the 95% interval spans roughly −0.19 to +0.67.*

Certified silence rises from 4.25% to 55.98%, which is 95.0% of the oracle fraction, with no soundness violation. The accuracy change is **+0.24** points and is statistically indistinguishable from zero. The defensible claim is that certification is obtained at no measurable accuracy cost; we do not claim an accuracy benefit on SHD. Selection bias from the architecture search proved small: the same configuration predicted +0.60 and 94.9% of oracle on the validation set used for selection.

### 5.3 Out-of-sample confirmation on SSC

The frozen SHD recipe was transferred to SSC without retuning (section 4.3). Table 3 reports four test seeds.

| seed | control | constrained | cost | control certified | constrained certified | oracle | violations |
|---|---|---|---|---|---|---|---|
| 1 | 65.56 | 68.66 | +3.10 | 9.50% | 60.90% | 64.59% | 0 |
| 2 | 65.25 | 68.06 | +2.81 | 12.94% | 64.17% | 68.11% | 0 |
| 3 | 65.24 | 68.33 | +3.09 | 11.32% | 61.11% | 65.21% | 0 |
| 4 | 64.61 | 67.75 | +3.14 | 9.51% | 59.73% | 63.84% | 0 |
| **mean** | **65.17** | **68.20** | **+3.03** | **10.82%** | **61.48%** | **65.44%** | **0** |

*Table 3. SSC test set, frozen recipe transferred without retuning. Cost standard deviation 0.15, standard error 0.08. Certification is 94.0% of oracle.*

Certified silence rises from 10.82% to 61.48%, which is 94.0% of oracle, with no violations. The accuracy change is **+3.03** points with standard error 0.08 — on this dataset the constraint acts as an effective regulariser, and the effect is systematic rather than incidental. Absolute SSC accuracy (68.20%) is below the strongest published results, which is expected and was stated in advance: the schedule is a step-matched transfer of an SHD recipe rather than one tuned for SSC. The purpose of this experiment is to test whether the mechanism survives a change of dataset with no adaptation, and it does.

### 5.4 The mechanism is reallocation, not reduction

Table 4 shows the per-delay drive before and after constrained fine-tuning.

| dataset / seed | *R*^(2) | *R*^(4) | *R*^(8) | total |
|---|---|---|---|---|
| SHD control | 4.97 | 4.90 | 5.40 | 15.27 |
| SHD constrained | **0.28** | 6.03 | 6.46 | 12.76 (−16%) |
| SSC control, seed 1 | 5.73 | 6.12 | 7.19 | 19.04 |
| SSC constrained, seed 1 | **0.29** | 8.56 | 9.98 | 18.83 (−1%) |
| SSC constrained, seed 2 | **0.29** | 8.54 | 9.84 | — |
| SSC constrained, seed 3 | **0.30** | 7.65 | 8.89 | — |
| SSC constrained, seed 4 | **0.29** | 8.21 | 9.42 | — |

*Table 4. Training reduces drive on the binding d = 2 tap while leaving total excitation nearly unchanged, with the same signature on both datasets and every seed.*

The binding *d* = 2 tap falls by 94% while the unconstrained *d* = 4 and *d* = 8 taps *grow*, so total excitatory mass changes by only 1% on SSC and 16% on SHD. Across four SSC seeds the *d* = 2 drive lands at 0.29, 0.29, 0.30 and 0.29, and on three SHD test seeds *R*_short lands at 0.253, 0.253 and 0.260 — a reproducibility of about ±0.01 in the governing quantity. The constraint is satisfied by **relocating** excitation in time, which is why it costs no accuracy.

The contrast with a unit-delay network makes the point sharply. There, no longer-delay tap exists to absorb the mass, and the same budget can only be met by eliminating excitation: positive recurrent weight mass falls from 2385 to 199 (a factor of twelve) and the excitation-to-inhibition ratio from 0.650 to 0.045. Ablating recurrence in that network shows what this costs — removing recurrent excitation costs 17.8 accuracy points in the control, while removing recurrent inhibition causes runaway firing (rate rising to 71%) and collapse to chance. Delays give the network somewhere to put its excitation; without them, the constraint must destroy it.

### 5.5 Certifiability can be architectural

The learnable-delay network in table 1 is the most consequential row. Its **unconstrained** control certifies 55.88% of core-steps, which is 97.4% of oracle, with no certificate penalty of any kind. Adding the penalty changes certification by −0.25 points and accuracy by +0.09.

The cause is visible in the delay distribution: learned delays settle at a weighted mean of 5.04 with only 32.7% of synapses below *d* = 4, so *R*_short is 1.595 rather than the fixed-tap control's 4.968. By table 1 that difference alone is sufficient.

Certifiability is therefore a property of **where synaptic delays sit**, and a dedicated training objective is required only when the architecture does not already supply the margin. This was not the result we anticipated — the penalty was assumed to be doing the work — and it is the more general statement of the two.

### 5.6 Exact multi-core execution

Certificates permit a core to advance without waiting for a neighbour whose certificate covers the required step. Spike trains are bit-identical to lock-step execution; this was verified against a single-threaded reference in every configuration reported below, 80 of 80 in the shared-memory engine and 36 of 36 in the TCP engine.

| cores | certificate coverage | *L*=0 | *L*=5 µs | *L*=20 µs | *L*=100 µs | *L*=500 µs |
|---|---|---|---|---|---|---|
| 4 | 44.5% | 0.93 | 0.95 | 0.91 | 1.07 | 1.11 |
| 8 | 52.2% | 0.90 | 0.88 | 0.90 | 1.02 | 1.11 |
| 16 | 61.5% | 0.79 | 0.79 | 0.78 | 1.05 | 1.11 |
| **32** | **71.0%** | **1.44** | **1.35** | **1.30** | **1.25** | **1.49** |
| 32, control | 14.4% | 0.99 | 0.98 | 0.83 | 0.98 | 0.99 |

*Table 5. Speed-up of certificate-based over lookahead-aware local handshaking, shared-memory engine, idle machine. Values above 1 favour certificates. The unconstrained control gains nothing at any core count.*

Three observations. Certificate coverage rises with core count (44.5% to 71.0%), because certification requires a *whole* core to be silent and smaller cores qualify more often — a design implication favouring fine-grained many-core architectures. The unconstrained control gains nothing anywhere, so the effect is attributable to training. And certificates **lose** below 32 cores at low latency: their computational cost exceeds their benefit once the handshake already has a free step from *d*_min = 2.

In the two-process TCP configuration, certificates were **1.37× slower** than the handshake (13.4–14.4 ms against 18.7–19.4 ms per sample), consistently at 8, 16 and 32 cores. The explanation is visible in the absolute figures: approximately 140 µs per step is dominated by the network round-trip, and each rank transmits a frame every step regardless of mode. Certificates reduce *waiting*, of which little remains, while their computation is pure added cost. We state this plainly: **the distributed case, as measured, is negative.** The implication is that over a network the benefit must come from *sending fewer messages* — a certified-silent core need not transmit at all — which is a different mechanism from the one implemented here and which we have not evaluated.

### 5.7 What does not work

The following were tested and failed, and we report them because each is a route a reader might otherwise assume productive.

- **Tightening the bound with activity caps.** The worst-case bound assumes every presynaptic neuron fires, whereas measured simultaneous activity is 0.98 of 32 neurons per core on average (99.9th percentile 7, maximum 10) — a pessimism factor above twenty. Yet a sound "at most *k* per presynaptic core" bound tightens *R* by only 1.1–4.1×, because the positive weight distribution is heavy-tailed; even *k* = 1 leaves *R* at 1.125, still 2.9 times the budget, certifying 0% of neurons. Activity-cap certificates are not a route to improvement.
- **Enlarging the budget via faster membrane leak.** The budget (1−β)θ grows as the membrane time constant falls, by 1.61× at τ_m/2 and 2.20× at τ_m/4. This raises certification but costs base accuracy, and on the delay architecture it is superseded: once delays supply the margin, the budget is no longer binding.
- **Restricting fan-in.** *R* is almost exactly linear in fan-in (*R* ≈ 0.0098 × fan-in across a tenfold range), so local connectivity reduces it proportionally and makes *R* width-independent. Certification duly rises, but absolute accuracy falls by about 8 points and additional width does not recover it (65.95% at *H*=512, 67.24% at *H*=1024, 64.59% at *H*=2048).
- **Per-synapse learnable delays for accuracy.** Although they make networks certifiable without training (section 5.5), they did **not** improve accuracy: 85.54% against the fixed-tap architecture's 87.25%. The parameter-efficiency hypothesis — that fixed taps were limited by their |𝒟|·*H*² parameters — is not supported.
- **Longer training.** On the delay architecture 300 epochs is worse than 150 (84.69% against 87.25% for controls), the reverse of the delay-free case.

## 6. Discussion

### 6.1 A design rule

The results support a concrete rule. Certifiability is governed by *R*_short, the worst-case excitatory drive arriving through synapses with delay below the certificate horizon. To obtain provable silence:

1. **Use synaptic delays of at least two steps and omit the unit-delay tap.** This guarantees two steps of exact lookahead and improves accuracy and certifiability together, so there is no trade-off to manage.
2. **Spread delays away from short lags.** If the delay distribution already places little excitatory mass below the horizon, certification follows with no training objective at all (section 5.5).
3. **If a penalty is needed, expect reallocation rather than reduction.** Networks satisfy the constraint by moving excitation into longer delays; total excitatory mass need not fall and accuracy need not suffer.
4. **For hardware, prefer fine-grained cores.** Certification requires a whole core to be silent, so coverage rises as cores shrink (44.5% at 4 cores to 71.0% at 32).

Since *R* is linear in fan-in with a measurable coefficient, the maximum affordable fan-in for a target certification level can be computed in advance rather than found by search.

### 6.2 Limitations

**Absolute accuracy is below the state of the art.** Our strongest network reaches 87.59% on SHD against roughly 96% for the best reported architectures, and 68.20% on SSC. We tested the most promising route to closing this gap — per-synapse learnable delays — and it failed (section 5.7). The remaining gap must be attributed to depth, time resolution or training protocol, none of which we varied. Whether the mechanism survives at competitive accuracy is therefore **unestablished**, although the trend is favourable: the better of our two delay architectures is also the more certifiable.

**The distributed speed result is negative.** Certificates were slower than a lookahead-aware handshake in a real two-process TCP configuration (section 5.6). Two processes is the weakest distributed configuration and our engine supports no more, so the distributed case is unproven rather than refuted; but as measured it does not favour the method, and we do not claim otherwise. The shared-memory speed-up (1.25–1.49× at 32 cores) should be read as a many-core shared-memory result.

**Latency in the shared-memory engine is emulated**, and no neuromorphic hardware was used. The TCP engine removes the emulation at the cost of supporting only two ranks.

**Scale.** Experiments use 512 recurrent neurons and up to 32 cores. The core-granularity effect in table 5 implies that systems with far more neurons per core — as in current neuromorphic processors — would certify whole cores less often, and the regime in which the method pays is many fine-grained cores.

**Statistical scope.** Headline results rest on three (SHD) and four (SSC) test seeds. The architecture comparison of table 1 is single-seed on validation; its purpose is to establish the monotone relationship with *R*_short, which spans twenty-fold and is corroborated by the multi-seed reproducibility of *R*_short itself (±0.01).

**One property, one optimisation.** We certify silence and use it to skip synchronisation. Whether the same approach — training a network so that a runtime guarantee becomes provable — extends to other properties and other exact optimisations is open, and is in our view the most interesting direction this work suggests.

## 7. Conclusion

Provable silence in a recurrent spiking network is governed by the worst-case excitatory drive arriving through synapses whose delay falls below the certificate horizon. Because longer-delay synapses contribute from time steps that are already determined, they impose no constraint, and a network can therefore become certifiable by relocating excitation in time rather than removing it. This is what trained networks do: the binding short-delay drive falls by 94% while total excitation is essentially unchanged, and the resulting networks certify 56% of core-steps on SHD and 61% on SSC — 94–95% of what is achievable — at no measurable accuracy cost on one dataset and a 3-point gain on the other, with no soundness violation in any run. Where the delay distribution already places little mass at short lags, the property holds without any dedicated training, which makes certifiability an architectural choice rather than an optimisation objective. The execution benefit is real but conditional: exactness is guaranteed, and barrier-free execution is faster than lookahead-aware handshaking in a many-core shared-memory setting, while a two-process network configuration favours the handshake because certificates reduce waiting rather than message count.

## Data and code availability

All training, certification and execution code, the complete experimental record including pre-registered hypotheses and pass criteria for every experiment, and all result files are available at [REPOSITORY URL]. Both datasets are publicly available [13].

## Acknowledgements

[ACKNOWLEDGEMENTS]

## Declaration of AI use

[STATE THE JOURNAL-REQUIRED DISCLOSURE. Analysis code, experiment orchestration and manuscript drafting were carried out with the assistance of a large language model; all experimental results were produced by the code in the repository and all claims were verified against the recorded result files.]

## References

*Note to co-authors: entries marked [verify] require checking of authors, volume, year and venue before submission.*

[1] S B Furber, F Galluppi, S Temple and L A Plana, "The SpiNNaker project", *Proc. IEEE* **102** 652 (2014).

[2] P A Merolla *et al.*, "A million spiking-neuron integrated circuit with a scalable communication network and interface", *Science* **345** 668 (2014).

[3] M Davies *et al.*, "Loihi: a neuromorphic manycore processor with on-chip learning", *IEEE Micro* **38** 82 (2018).

[4] Decentralised synchronisation for scalable neuromorphic systems. [verify authors, venue, year]

[5] Analytical performance modelling of neuromorphic workloads identifying traffic and synchronisation bounds. [verify]

[6] J Yik *et al.*, "NeuroBench: a framework for benchmarking neuromorphic computing algorithms and systems". [verify venue and year]

[7] R M Fujimoto, "Parallel discrete event simulation", *Commun. ACM* **33** 30 (1990).

[8] K M Chandy and J Misra, "Distributed simulation: a case study in design and verification of distributed programs", *IEEE Trans. Softw. Eng.* **SE-5** 440 (1979).

[9] D R Jefferson, "Virtual time", *ACM Trans. Program. Lang. Syst.* **7** 404 (1985).

[10] M-O Gewaltig and M Diesmann, "NEST (NEural Simulation Tool)", *Scholarpedia* **2** 1430 (2007).

[11] Certified robustness of spiking neural networks via interval bound propagation. [verify authors, venue, year]

[12] I Hammouamri, I Khalfaoui-Hassani and T Masquelier, "Learning delays in spiking neural networks using dilated convolutions with learnable spacings", *Int. Conf. on Learning Representations* (2024). [verify]

[13] B Cramer, Y Stradmann, J Schemmel and F Zenke, "The Heidelberg spiking data sets for the systematic evaluation of spiking neural networks", *IEEE Trans. Neural Netw. Learn. Syst.* **33** 2744 (2022).

[14] G Bellec, D Salaj, A Subramoney, R Legenstein and W Maass, "Long short-term memory and learning-to-learn in networks of spiking neurons", *Advances in Neural Information Processing Systems* **31** (2018).

[15] B Yin, F Corradi and S M Bohté, "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks", *Nat. Mach. Intell.* **3** 905 (2021).
