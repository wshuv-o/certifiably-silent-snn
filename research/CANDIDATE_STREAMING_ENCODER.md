# Candidate Direction — Event-Driven Incremental Encoding for Streaming Speech (v0, 2026-10-04)

Status: **candidate, not adopted**. Feasibility pilot pre-registered below. Software-only; runs on the RTX 2060.

## Real-world scenario
Live captioning and transcription on laptops and phones with off-the-shelf Whisper (meetings, accessibility, lectures).

## Gap (from the newest systems' own descriptions)
- **WhisperFlow / Whisper-T** ([arXiv:2412.11272](https://arxiv.org/html/2412.11272v1)): the encoder **re-encodes the full overlapping buffer every round**. The hush word, beam pruning and pipelining all work inside that cost.
- **NPUsper** ([arXiv:2607.01108](https://arxiv.org/html/2607.01108), 2026): avoids padding by processing short inputs "with minimal carryover". It **drops context rather than reusing computation**.
- **Causal / streaming conversions** (WhisperRT [arXiv:2508.12301](https://arxiv.org/pdf/2508.12301); two-pass U2 [arXiv:2506.12154](https://arxiv.org/html/2506.12154v1)) need **retraining** and change the model.
- Not found (two searches): **training-free** reuse of encoder computation across overlapping windows of a *bidirectional* encoder.

## Brain-inspired principle
Old frames behave like neurons that fire only when their input changes. Their acoustic input is identical across rounds; they change only through attention from newly appended frames.

## Mechanism sketch (to be designed only if the pilot passes)
1. **Attention-mass event gating.** Per layer, predict each old frame's update from the attention mass it places on changed keys (bounded by mass × value change). Recompute only frames above a threshold; reuse cached activations for the rest.
2. **Position handling.** The buffer grows (prefix positions fixed) until it is trimmed, which shifts absolute positions. Needs a strategy (aligned trimming or a re-anchoring cost model).
3. **Bounded deviation** from full recomputation, checked by WER and an encoder-output error bound.

## Distinction from Eventful Transformers (ICCV 2023), the expected reviewer objection
- Eventful gates tokens by **change in their own input** (video frames change everywhere).
- Here old inputs are **constant**; change arrives only via attention propagation, so it can be **predicted before computing** from attention mass.
- Also: sliding-window position shifts, and an encoder–decoder (cross-attention) consumer.

These differences must be shown to matter experimentally, or the objection stands.

## PILOT-001 (PRE-REGISTERED 2026-10-04, before running): is there reusable structure?
- **Setup:**
  - Whisper (HF transformers), sizes `base` and `small`.
  - LibriSpeech utterances (pilot: HF dummy subset, concatenated to ≥ 20 s).
  - Buffer = first T seconds; append Δ = 1 s.
  - Compare encoder hidden states of the overlapping frames between the T-second and the (T + Δ)-second inputs, per layer.
  - Use **the same padding convention as Whisper (30 s pad)** and also **no padding**.
- **Measures:**
  - Per layer: relative change ‖h' − h‖ / ‖h‖ for each old frame, as a function of the frame's distance from the boundary.
  - Fraction of old frames below 1% relative change at the final encoder layer.
- **Viability criterion (fixed now):**
  - VIABLE if, at the final encoder layer, at T = 10 s, Δ = 1 s, **≥ 60% of old frames change by < 1%**, and change decreases with distance from the boundary.
  - KILL if the median relative change of old frames at the final layer is **> 5%** (global mixing: nothing reusable).
  - In between: inconclusive. Test whether decoding (WER) tolerates freezing far frames before deciding.
- Note: with 30 s padding the input contains padding frames whose content changes when audio is appended. That is itself an important structural effect to record.

## PILOT-001 — RESULT (2026-10-04): premise FAILED. Candidate not adopted.

- **Verification:** manual encoder forward vs HF on a 30 s input, max abs diff = 0.0 (base and small).
- **Data:** HF librispeech_asr_dummy (73 utterances, concatenated), 3 segments × T ∈ {5, 10, 20} s, Δ = 1 s, pad30 and nopad, whisper-base and whisper-small, CPU. File: `results/pilot001.json`.
- **Final encoder output, T = 10 s** (mean over 3 segments):

| model, padding | frames with change < 1% | median change | change 0–1 s from boundary | 5–10 s from boundary |
|---|---|---|---|---|
| base, nopad | 0.3% | 4.1% | 20.2% | 3.4% |
| base, pad30 | 0.1% | 3.8% | 14.6% | 3.3% |
| small, nopad | 0.1% | **5.0%** | 19.9% | 4.3% |
| small, pad30 | 0.0% | 4.6% | 16.4% | 3.9% |

- **Per-layer median change** (T = 10, nopad): base rises from 0.8% after layer 1 to 4.1% at the output; small rises from 0.75% to 5.0%. Change **accumulates through depth**.
- Longer buffers dilute the change (T = 20: median 1.8–3.1%), but frames under 1% are still only 0–2.4%.
- **Scoring against the pre-registered criteria:**
  - VIABLE (≥ 60% of frames < 1%): **not met in any condition** (max 2.4%).
  - KILL (median > 5%): met for small/nopad (5.01%); the other conditions are in the inconclusive band (3.8–4.6%).
- **Conclusion:** the core premise — that most old frames are unaffected by new audio, so their computation can be skipped — **is false for Whisper**. Bidirectional attention spreads a few-percent change to essentially every frame, and it grows with depth.
  - The only remaining route is an *approximate* version: tolerate stale frames and check whether WER survives. Even if WER survived, that would be "streaming with stale cached context" without retraining, a weaker and less novel claim than exact event-driven reuse.
  - **Decision: candidate dropped.** Recorded as a negative result.
- **Lesson for the next brainstorm:** brain-inspired "skip unchanged units" needs a model whose units really are unchanged. Deep bidirectional attention models violate that. Check this property with a pilot like this one *before* building any mechanism.
- **Disk note:** model downloads grew the WSL disk; D: free fell to 4.2 GB.
