# Handoff note — continuing on the RTX 5080 machine (written 2026-10-06)

## 1. What this project is
**Certifiably Silent Spiking Networks.** We train recurrent SNNs so that their silence is *provable*. Each neuron's worst-case excitatory recurrent drive R_i is kept below the budget (1−β)θ; per-neuron certificates (Lemma 1/2) then prove "no spike for K steps", and multi-core execution can skip waiting with bit-identical results.

- **Manuscript:** `research/paper/manuscript.md` (source) and `manuscript.docx` (built). Figures are in `research/paper/figures/`.
- **Lab notebook (every experiment, pre-registered, with outcome):** `research/N3_SCALEUP_PLAN.md`. Read its end first.
- **Research rules:** `research/RESEARCH_OS.md` (incl. the measurement-trap guardrail). Status: `research/PROJECT_BRIEF.md`.

## 2. Where things stand (2026-10-06 ~11:45)
**Confirmed:**
- certificates are sound (0 violations in > 317 M certified windows, incl. 8-bit integer arithmetic);
- every excitation-bounding method certifies ≈ 60% of core-steps (84–99% of actual silence);
- sparsity ≠ certifiability (rate penalty: 0.8%);
- exact speed-up 1.5–2.3× for locally connected cores at ≥ 20 µs latency.

**Open problem:** the accuracy cost on the 512-ALIF model is ≈ 2 points on the unseen-speaker test set.
- ExCap (hard excitation cap) won SWEEP-001 on validation but **failed** test confirmation (−4.5 points).
- Cause: the random validation split had the same speakers as training.

**In flight on the RTX 2060 when this was written — SWEEP-002** (accuracy improvement, pre-registered):
- speaker-disjoint validation (held-out training speakers 3 and 6);
- fine-tuning from the trained control (FT) and knowledge distillation (KD).
- If it did not finish there, rerun it on the 5080 (section 4).

## 3. Setup on the 5080 machine (Linux or WSL2 Ubuntu 24.04)
```bash
git clone https://github.com/wshuv-o/certifiably-silent-snn.git
cd certifiably-silent-snn
bash setup_5080.sh            # creates ~/research/{venv_gpu,venv_speech,data,models,build}
```
- **RTX 5080 = Blackwell (sm_120).** It needs PyTorch built for **CUDA 12.8+** (`--index-url https://download.pytorch.org/whl/cu128`). The setup script does this. Older cu126 wheels will not run on it.
- **Datasets download automatically** on first use (SHD ≈ 0.35 GB; SSC ≈ 3.4 GB, only for `DATASET=ssc`) into `~/research/data/`.
- **Trained models are not in git** (~/research/models). Retrain them with the scripts. Everything is seeded.
- **C++ engines** need `g++` (C++20) and `-pthread`. The setup script installs `build-essential`.

## 4. How to run (all scripts live in `harness/`, run from there)
| Goal | Command |
|---|---|
| SWEEP-002 stage 1 (control/teacher + reference) | `bash run_sw2_stage1.sh` |
| SWEEP-002 stage 2 + automatic winner selection | `bash run_sw2_auto.sh` |
| Confirm the winner on seeds 2–4 (test set) | `bash run_confirm2.sh` |
| Speed test (C++ engine, needs an idle CPU) | `bash run_engine_final.sh` (edit model names to the confirmed winner) |
| Figures | `bash run_figs.sh` |
| Build the Word manuscript | `python md2docx.py ../research/paper/manuscript.md ../research/paper/manuscript.docx` |

**Using the 5080's 16 GB:**
- In `run_sw2_stage2.sh` the configs run in 2 parallel queues; split them into 4–6 queues (each run uses ≈ 1–1.5 GB).
- `run_confirm2.sh` already runs seeds 2–4 in parallel.
- Optional extra speed: `torch.autocast("cuda", dtype=torch.bfloat16)` around the forward pass, and/or `torch.compile`. **Check the results do not change materially before trusting them.**

## 5. Rules we follow (keep following them)
1. **Pre-register** every experiment (hypothesis + pass/fail) in `research/N3_SCALEUP_PLAN.md` *before* running.
2. **Never select on the test set.** Select on speaker-disjoint validation; confirm on fresh seeds on test.
3. **Report every outcome**, including failures.
4. **Run our method's jobs first, then baselines** (unless ours needs a trained baseline as init/teacher).

## 6. Next tasks (in priority order)
1. Finish SWEEP-002 + CONFIRM-002 → decide the final training recipe.
2. Speed test of the confirmed recipe on an idle CPU (`run_engine_final.sh`).
3. Update the manuscript (results section 5.3/5.4, Appendix A), rebuild the docx, re-render the figures.
4. With the 5080's capacity:
   - more seeds for the single-seed results;
   - SSC with 40+ epochs;
   - fan-in-aware budgets (cost grows with network size).
5. Before submission:
   - verify the references marked "[verify]";
   - add author info and the AI-use statement.
