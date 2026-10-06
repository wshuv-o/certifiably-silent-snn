# Certifiably Silent Spiking Networks

Bounding excitatory recurrent drive makes recurrent SNN silence **provable**. Silence certificates then let multi-core neuromorphic execution skip synchronization with **bit-identical** results.

- **Paper draft:** `research/paper/manuscript.docx` (source: `manuscript.md`)
- **Experiment log** (pre-registered, every outcome reported): `research/N3_SCALEUP_PLAN.md`
- **Code:** `harness/` — training (`s2_strong.py`, `s3_sweep.py`, `s4_improve.py`, `pilot_silence.py`), certificate checks, fixed-point verification (`fxp_check.py`), exact C++ multi-core engines (`s1b_engine.cpp`, `s3_engine.cpp`), figures (`make_figures.py`)
- **Continuing on a new GPU machine:** see `HANDOFF_5080.md` and `setup_5080.sh`

Private research repository — unpublished work.
