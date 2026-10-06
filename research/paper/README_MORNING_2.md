# Morning summary — overnight of 2026-10-06/07

## The short version

**The project's central problem is solved, and it is confirmed on the test set with fresh seeds and on a
second dataset with zero retuning.** Provable silence now costs **nothing** — it slightly *gains*
accuracy. The design rule that achieves it was **derived from theory before it was measured**, and the
same mechanism signature appears on both datasets.

| | SHD (test, frozen recipe, seeds 2-4) | SSC (test, frozen, no retuning) |
|---|---|---|
| control certified | 4.2% | 9.50% |
| **constrained certified** | **55.98%** | **60.90%** |
| % of oracle | **95.0%** | **94.3%** |
| **accuracy cost** | **+0.24** (CI about [-0.19, +0.67]) | **+3.10** |
| test accuracy | **87.59%** (was ~80%) | 68.66% |
| violations | **0** | **0** |

## What made the difference: synaptic delays, and a decomposition

With `v(t+1) = BETA*v(t) + I(t+1) + sum_d W_d s(t+1-d)`, a certificate at horizon `k` needs spikes from
`t+1+k-d`. **For `d >= k` those are already past — known exactly; only `d < k` must be bounded.** So only
taps with `d < K` bind a K-step certificate, and the network can satisfy the budget by **moving**
excitation into the free long taps rather than destroying it.

That came out of the algebra before any run. Both datasets then did exactly it:

| R_per_delay | d=2 | d=4 | d=8 | total |
|---|---|---|---|---|
| SHD control | 4.97 | 4.90 | 5.40 | 15.27 |
| SHD constrained | **0.28** | 6.03 | 6.46 | 12.76 (-16%) |
| SSC control | 5.73 | 6.12 | 7.19 | 19.04 |
| SSC constrained | **0.29** | 8.56 | 9.98 | **18.83 (-1%)** |

The binding d=2 tap collapses to ~0.28 on **both** datasets, just under the 0.3935 budget, while the free
taps grow and total excitation is nearly unchanged. On the three SHD confirmation seeds it lands at
0.253, 0.253, 0.260 — the mechanism reproduces, it is not a fluke.

**Contrast the old single-delay architecture:** the same budget forced destroying 92% of recurrent
excitation (E/I ratio 0.650 -> 0.045) and cost accuracy. **Delays give the network somewhere to put its
excitation**, so the constraint becomes a *reallocation* instead of a *reduction*.

**Design rule for the paper:** use delays >= 2 and omit the unit-delay tap. It improves accuracy *and*
certifiability at once — there is no trade-off on this architecture.

## The speed result is weaker, and that is the honest headline

On an idle CPU, with the handshake baseline **given the free lookahead the delays provide** (anything else
would be a strawman):

| cores | L=0 | L=5 | L=20 | L=100 | L=500 | coverage |
|---|---|---|---|---|---|---|
| 4 | 0.93 | 0.95 | 0.91 | 1.07 | 1.11 | 44.5% |
| 8 | 0.90 | 0.88 | 0.90 | 1.02 | 1.11 | 52.2% |
| 16 | 0.79 | 0.79 | 0.78 | 1.05 | 1.11 | 61.5% |
| **32** | **1.44** | **1.35** | **1.30** | **1.25** | **1.49** | 71.0% |
| control (32) | 0.99 | 0.98 | 0.83 | 0.98 | 0.99 | 14.4% |

- Certificates **lose** at 4–16 cores at low latency: their runtime cost exceeds their benefit once the
  handshake has a free step from `d_min = 2`.
- They win only at **32 cores** (1.25–1.49x). The control never wins, so that gain is entirely
  training-attributable.
- Against the single-delay architecture's **1.67–1.71x**, this is **lower** — because delays made the
  baseline stronger. **Delays buy accuracy and certifiability at the cost of the certificate's marginal
  speed advantage.** I pre-registered this expectation before measuring.
- Caveat: 32 cores on a 32-thread machine saturates the scheduler (medians jump to 39–79 ms from 4–8 ms
  at 16 cores), which plausibly inflates the benefit of waiting less. Do not extrapolate without
  re-measuring on a machine with spare threads.

## What is solid

- **0 soundness violations** in every run, dataset, width and seed, ever.
- The delay-aware exact engine is **bit-identical to a single-thread reference in every configuration**
  (4/8/16/32 cores x 5 latencies x 2 modes x 2 models, twice over).
- Certificate coverage **rises with core count** (44.5% -> 71.0%); the control gets 0.4–0.8% below 32
  cores, so **training supplies essentially the whole effect**.
- `R ~ 0.0098 x fan-in`, linear across a 10x range — a predictive design formula.
- Validation-selection bias, which I flagged as the biggest risk, proved **negligible**: validation
  predicted +0.60/56.28%/94.9%, test delivered +0.24/55.98%/95.0%.

## What is still missing

1. **No real hardware.** Latency is emulated. Two machines over real Ethernet is days of work and would
   remove the single most damaging objection.
2. **Sub-SOTA accuracy.** 87.59% on SHD vs ~96%; 68.66% on SSC vs ~80%. My multi-tap delays multiply
   weights by the tap count; the ~95% methods learn **one delay per synapse** at the same weight budget.
   E1 proved the point by collapsing to 73.74% at 4.19M parameters — the model is data-limited at ~1M.
3. **SSC is seed 1.** The SHD result has three seeds; SSC needs the same.
4. **The speed story is now architecture-dependent** and the paper must state which architecture each
   number belongs to.

## Mistakes I made and corrected (all logged)

- Reported "no measurable cost" from 3 seeds; six seeds gave -0.91 with a CI excluding zero. Rule adopted.
- Compared a seed against a **pooled** control mean instead of its own control. Rule adopted: always
  seed-matched.
- Read a mid-schedule cosine-LR number and concluded more epochs bought nothing; it gained +4.02 by the
  end. Rule adopted.
- Pushed GPU concurrency to 6 processes at your request; it gave **zero** throughput gain and destroyed
  9 runs via CUDA faults. Reverted to 3 with retry + resume.
- Benchmarked the engine twice on a contaminated CPU, the second time via a race in my own idle-check
  that sampled the one-second gap between two queued jobs. Both runs discarded; gate now requires an
  explicit completion marker plus five consecutive idle checks.

## Suggested next steps, in order of value

1. **Per-synapse learnable delays** (DCLS-style) — the one change that attacks the SOTA gap without
   multiplying parameters. 1–2 weeks.
2. **Two-machine real-network run** — deletes "emulated" from the central claim. Days.
3. **SSC seeds 2–4** — cheap, and the SSC result currently rests on one seed.
4. Re-measure the 32-core speed-up on a machine with spare hardware threads.

All work is committed. Full experimental record with every pre-registration, pass/fail bar and failure:
`research/N3_SCALEUP_PLAN.md`. Current status: `research/PROJECT_BRIEF.md`.
