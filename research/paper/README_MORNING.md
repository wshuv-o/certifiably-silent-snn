# Morning summary — overnight work on the N3 paper (2026-10-06, finished 07:15)

## Your deliverable
| File | What it is |
|---|---|
| `research/paper/manuscript.docx` | **The full manuscript** (16 pages, ~5,500 words, 6 figures, 4 tables, proofs, Appendix A with every pre-registered result) |
| `research/paper/manuscript.md` | The source text (edit this, then rebuild with `harness/md2docx.py`) |
| `research/paper/figures/` | 6 figures, 300 dpi, colour-blind-checked |
| `research/N3_SCALEUP_PLAN.md` | Lab notebook: every experiment, its pre-registration and its outcome |
| `harness/` | All code: training, certificates, fixed-point check, C++ engines, figures |

**Title:** *Certifiably Silent Spiking Networks: Bounding Excitation Makes Recurrent SNN Silence Provable and Reduces Synchronization in Exact Multi-Core Execution*

## What the paper claims — all backed by pre-registered experiments
1. **The problem:** trained recurrent SNNs are silent ~60% of the time, but 0–1% of that is provable, so hardware cannot safely skip synchronization.
2. **Certificates with proofs** (Lemma 1, Lemma 2, adaptive-threshold and fixed-point extensions). **0 violations in > 158 million certified windows.**
3. **The core principle:** provability is governed by one quantity, the worst-case excitatory drive R_i relative to the budget (1−β)θ (Proposition 1). The margin predicts certified horizon (ρ = 0.83–0.87).
4. **Any method that bounds excitation works** (certificate loss, L1, hard cap): 56–62% provable silence (84–98% of actual) on every model and dataset tested.
5. **Sparsity ≠ certifiability:** a strong firing-rate penalty gives the sparsest networks but certifies only 0.8%. This is a clean, memorable result.
6. **Exact speed-up:** 1.5–2.3× over local handshaking for locally connected cores at 20–500 µs latency, with bit-identical spikes.

## What changed overnight (be aware)
- **The framing shifted (pre-registered rule b).** Simple L1 or a cap works as well as our special loss. The contribution is the *principle plus certificates plus exact execution*, not one loss function. This is arguably stronger: a one-line recipe practitioners can use.
- **Honest limitations now in the paper:**
  - Accuracy cost is negligible on SHD up to 512 neurons, but **grows with size** (−2.1 points at 1,024, 3 seeds) and is **−3 to −3.7 points on SSC** (second dataset, 15 epochs).
  - The **dense-connectivity speed-up is only 1.15–1.40×**.
  - Latency is **emulated** (no neuromorphic hardware).
  - The best accuracy (84%) is **below SOTA** (~96% on SHD).
- **Process issues, all disclosed:** one invalid engine run (thread oversubscription, rerun), one overwritten result file (seed-1 values taken from logs), one bad certificate design in S1b (fixed in S1c).

## Is it Q1? Honest assessment
- **Realistic Q1 targets:** *Neural Networks*, *Neurocomputing*, *Neuromorphic Computing and Engineering* (check current quartiles on Scimago/JCR).
- **Strengths reviewers will like:** a clear principle with proofs; a surprising negative control (sparsity ≠ certifiability); exactness; real (emulated) speed-ups; unusually transparent pre-registration with failures reported.
- **Main objections to expect:**
  1. No real hardware.
  2. Accuracy below SOTA, and the cost grows with size.
  3. Modest speed-ups, smaller with dense connectivity.
  4. Single seeds for some results.
- **IEEE TNNLS / Nature-family:** would likely need real multi-chip hardware (e.g. SpiNNaker2) and SOTA-level models.
- **I cannot promise acceptance.** But this is now a coherent, novel, honestly reported method paper with a realistic shot at a good Q1 journal.

## What you need to do (about 1–2 hours)
1. Fill in author name(s), affiliation and email; pick the journal and apply its template.
2. **Verify the references marked "[verify]"** (a few authors, volumes and venues).
3. Add the AI-use statement your journal requires (most require disclosure).
4. **Read the paper once end to end.** You must be able to defend every claim; you are the author.

## Highest-value next steps (if you want to strengthen it before submitting)
1. **Seeds for the single-seed results** (trade-off curve, S2b, SSC): a few GPU-hours.
2. **Longer SSC training** (40+ epochs), to see whether the 3–4 point cost shrinks.
3. **Fan-in-aware budgets**, to address the cost growth at 1,024 neurons. This is a natural method extension.
4. **SpiNNaker2 access:** a real multi-chip demonstration would lift the paper's ceiling the most.
