# Hardware measurements (section 5.7, table 7, supplement)

Code: `harness/hw/`. Routed-design reports: `harness/hw/build/pnr_*/` (flow.log, drc.rpt, synth.stat).
Cycle-level co-simulation logs: `results/hw001/`, `results/hw008/`.

**Reference model.** Fixed-point golden model (`golden.py`) against the floating-point engine, 5 samples:
spike agreement 0.99991 (local K=4 network), 0.99894 (DCLS host layer); 0 violations (`run_check.sh`).

**Cycle-level verification.** 64/64 runs (4 networks x 16 cores x 3 samples) for each core version, zero mismatches
in spikes, certificate bits and membrane/adaptation state (`run_tb.sh`, `run_tb_lp.sh`, `sweep_tb*.sh`).
Core-step fractions normal / catch-up / idle: local K=4 0.583/0.139/0.278, dense 0.605/0.132/0.263,
DCLS host 0.676/0.108/0.216, SpikCommander host 0.653/0.116/0.231.

**Early abort** (`early_abort.py`): a normal pass uses stages 2..K for 0.48-0.51 of its neurons.

**Reduced-precision certificate** (`lowcert.py`, `lowprec.py`), directed rounding, saturating:
operands in units of 2^-8 with 6-bit decay retain 0.988 (local K=4) / 0.960 (DCLS host) of full-precision certification;
units of 2^-4: 0.769 / 0.573; no unsound certificate at any setting.
RTL against the reference: 0 mismatches on 16,000 vectors from the networks and 4,000 random vectors.

**Routed units** (sky130_fd_sc_hd, typical corner, 25 C, 1.8 V, 24 ns clock, DRC 0):

| unit | cell area (um2) | slack (ns) | power at measured mean density | energy |
|---|---|---|---|---|
| neuron unit | 30,320 | +5.71 | 0.908 mW at 0.154 | 22 pJ per update |
| certificate unit K=4, 16-bit | 49,381 | +9.12 | 1.50 / 1.42 mW at 0.195 / 0.182 | 36 / 34 pJ per evaluation |
| certificate unit K=4, 24-bit | 131,332 | +6.74 | 4.70 mW at 0.154 | 113 pJ per evaluation |

Power is the routed design's power at each design's measured mean data-net toggle density (Icarus gate-level
simulation of 5,000 operand cycles from the trained networks, `activity.py`, `act_low.py`, `power.sh`, `pmean.sh`);
energy = power x 24 ns.
