# Brainstorm 002 — Brain-Inspired Mechanisms for Real-Life Laptop AI (2026-10-04)

- **Method:** from knowledge only. **No web search** (requires user permission), so every "nearby work" entry is *from memory and unverified*, and every novelty judgement is provisional.
- **Constraints:** software only; RTX 2060 6 GB plus a 6-core CPU; ~5 GB free disk (small models only); start small.
- **Filter applied to each idea:**
  1. Real-life user.
  2. Brain principle doing real work, not decoration.
  3. **Key premise** testable with a ~1-hour pilot.
  4. Measurement-trap check.
  5. Likely collisions.
- **Lesson from PILOT-001:** "skip unchanged units" fails when deep attention spreads change everywhere. Every premise below must be checked by a pilot first.

---

## A. Edit-aware KV-cache reuse for local coding assistants ("memory reconsolidation")

- **Scenario:** a local code assistant (small LLM on a laptop) keeps a long file or repo context. When the user edits one line in the middle, standard prefix caching keeps only the KV *before* the edit; **everything after the edit is recomputed**. This happens on every keystroke burst.
- **Brain principle:** reconsolidation. Recalling and changing a memory updates the affected trace locally rather than rebuilding the whole memory.
- **Mechanism idea:** after an edit, recompute KV only for (i) the edited span and (ii) downstream tokens whose cached state is predicted to change beyond a tolerance. For a causal model, the prediction could use attention mass on the edited span; this is the same idea PILOT-001 tested, but here attention is *causal* and edits are *local*. Reuse the rest, with position re-indexing when the edit changes length (RoPE allows exact re-rotation).
- **Key premise (pilot):** after a small, realistic code edit, most downstream tokens' K/V change by < 1–2% in a small causal LLM. Equivalently, reusing stale K/V for low-change tokens leaves next-token predictions (top-1 agreement, perplexity) essentially unchanged.
- **Why the premise may hold here, unlike Whisper:**
  - Causal attention means tokens *before* the edit are exactly unchanged.
  - Code has strong local structure.
  - Positional shifts are exactly correctable under RoPE.
- **Nearby work (from memory, unverified):** CacheBlend (reuse of non-prefix RAG chunk KV with selective recomputation of a minority of tokens); prompt/prefix caching in vLLM and llama.cpp; Block-Attention.
- **Collision risk:** MEDIUM–HIGH. CacheBlend-style selective recomputation is the closest. A distinct angle would need to be the *edit* setting: repeated local edits, length-changing edits, an exact RoPE shift, and error that accumulates over many edits.
- **Impact:** high if it works. Copilot-style local assistants re-prefill long contexts constantly.
- **Pilot cost:** Qwen2.5-0.5B (~1 GB download), local Python files as code context (no dataset download), CPU or GPU, ~1 hour.

## B. Hebbian fast-weight memory for on-device personalization without backprop

- **Scenario:** personalize a frozen small model (keyboard prediction, local assistant, ASR vocabulary) to one user on a laptop, with no fine-tuning, no gradients and little memory.
- **Brain principle:** fast Hebbian synaptic plasticity stores recent associations next to slow, consolidated weights.
- **Mechanism idea:** a small associative memory attached to one layer, updated with a local Hebbian/delta rule from the user's own text, gated by surprise (prediction error). It adds a correction to the residual stream at inference.
- **Key premise (pilot):** on a per-author text split, a surprise-gated Hebbian memory measurably lowers perplexity on that author's held-out text (e.g., ≥ 5% relative) versus the frozen model, at < 1% extra compute. Plain kNN-LM-style retrieval is the baseline it must beat or match more cheaply.
- **Nearby work (from memory, unverified):** fast weights (Ba et al.); kNN-LM; memory layers; test-time training; Titans (surprise-based neural memory); MeZO (backprop-free fine-tuning).
- **Collision risk:** MEDIUM–HIGH. The specific "laptop personalization, no backprop, surprise-gated" combination may be open, but Titans and kNN-LM are close.
- **Impact:** medium–high (privacy-preserving personalization).
- **Pilot cost:** small LM (~0.5 GB) plus public per-author text; ~1–2 hours.

## C. Change-driven CNN inference for webcam effects (habituation per region)

- **Scenario:** background blur and segmentation in video calls run a CNN on every frame, although most of a talking-head frame is static.
- **Premise:** likely TRUE. CNNs have local receptive fields, so unchanged regions give exactly unchanged features (PILOT-001 confirmed this for Whisper's conv layers: 0.0 change).
- **Nearby work (from memory):** DeltaCNN (CVPR 2022), Skip-Convolutions (CVPR 2021), CBinfer.
- **Collision risk: HIGH**, and the gain is small (these models already run in a few ms). **Rejected at brainstorm.**

## D. Predictive-coding cascade for always-on assistants (run the big model only on surprise)

- **Scenario:** an always-on mic or camera assistant on a laptop runs a big model continuously.
- **Mechanism:** a tiny predictor forecasts the next input or embedding; the big model runs only when the prediction error spikes.
- **Nearby work:** VAD gating, NoScope/Focus-style video-analytics cascades, early-exit networks.
- **Collision risk: HIGH**, and the brain framing adds little over existing cascades. **Rejected at brainstorm.**

## E. Activity-history-driven weight offloading for LLMs on 6 GB GPUs (synaptic "warmth")

- **Scenario:** a model larger than VRAM runs with CPU offloading. Keep "hot" neurons or experts on the GPU, predicted from recent activation history.
- **Nearby work (from memory):** PowerInfer, Deja Vu, LLM-in-a-flash, MoE offloading caches.
- **Collision risk: HIGH** (a very active area). **Rejected at brainstorm** unless the literature check reveals a specific gap.

---

## Ranking

| Idea | Premise likely true? | Impact | Novelty plausibility (unverified) | Laptop pilot |
|---|---|---|---|---|
| **A. Edit-aware KV reuse** | plausible (causal + local + exact RoPE shift) | high | medium (CacheBlend is the threat) | yes, ~1 h, ~1 GB |
| **B. Hebbian personalization** | uncertain | medium–high | medium (Titans, kNN-LM are threats) | yes, ~1–2 h |
| C, D, E | — | — | low | rejected |

## Recommended next step

1. **Pilot A's premise first.** No web needed; ~1 GB model download.
   - If tokens after a code edit mostly change by < 1–2% (and stale reuse keeps predictions), the idea is alive.
   - If not, drop it as PILOT-001 was dropped.
2. **Only if the pilot passes:** ask for permission to run a short literature check (~5–8 searches) on CacheBlend and edit-time KV reuse before designing anything.
3. B is the backup.

## PILOT-002 — premise test for A (PRE-REGISTERED 2026-10-04, before running)

- **Model:** Qwen/Qwen2.5-0.5B (causal, RoPE), float32 on CPU.
- **Contexts:** 3 real Python source files from the local venv (numpy, pandas, transformers internals), first 2048 tokens each.
- **Edits** (at a line boundary near token 1024, so suffix tokenization is identical; checked automatically and skipped if not):
  1. rename an identifier on one line;
  2. change a numeric literal;
  3. insert a new line;
  4. delete a line.
- **Measures:** for suffix tokens (after the edit), per layer, the relative change of hidden states between the original and edited contexts, aligned by suffix offset (hidden states are pre-RoPE, so positional shifts don't contaminate the comparison). Reported against distance from the edit. Also: top-1 next-token agreement between the original and edited runs for suffix tokens.
- **Criteria** (last layer, suffix tokens more than 32 tokens after the edit, averaged over files × edits):
  - **VIABLE:** ≥ 60% of tokens change by < 1%, **and** top-1 agreement ≥ 98%.
  - **KILL:** median change > 5%.
  - Otherwise inconclusive.
