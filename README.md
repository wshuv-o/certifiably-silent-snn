# Short-Delay Excitatory Drive Limits Provable Silence in Recurrent Spiking Networks

Code, trained models, result files and manuscript source for the paper

> M W Suva, M T Islam, E M C Abha, M I A Sajin and M F Mridha,
> *Short-Delay Excitatory Drive Limits Provable Silence in Recurrent Spiking Networks*,
> submitted to *Neuromorphic Computing and Engineering*.

In multi-core neuromorphic systems a core waits for its neighbours' spikes before each time step and may skip the
wait only if their silence is proven in advance. For recurrent networks of adaptive leaky integrate-and-fire neurons
with multi-tap synaptic delays, a *K*-step silence certificate needs a worst-case bound only for synapses with delay
shorter than *K*. The repository contains the training code with the certificate penalty, the certificate and
oracle analyses, the multi-core execution engines, a register-transfer core with the certificate unit routed in
SkyWater 130 nm, the deployment on SpiNNaker, and the trained networks behind the main results.

## Layout

| path | contents |
|---|---|
| `harness/` | training (`s5_delays.py`, `s7_dcls.py`, `s4_improve.py`, `s2_strong.py`, `pilot_silence.py`), certification and analyses, run drivers |
| `harness/s6_engine_delays.cpp`, `harness/s8_engine_net.cpp` | shared-memory and TCP execution engines (handshake, certificate and oracle modes) |
| `harness/hw/` | Amaranth register-transfer core and certificate unit, fixed-point golden model, co-simulation, OpenROAD place-and-route, power and layout rendering |
| `harness/spinnaker/` | SpiNNaker deployment: network package, EBRAINS notebooks, analysis |
| `models/` | trained networks (PyTorch state dictionaries) |
| `results/` | result files underlying every table and figure (`results/records/` holds measurements recorded as tables) |
| `research/paper/` | manuscript and supplement (LaTeX source, figures, PDFs) |

## Trained models

| file | network | paper |
|---|---|---|
| `models/shd/local_K4_control.pt`, `local_K4_certified.pt` | locally connected, H=512, D={2,4,8}, without / with certificate penalty | tables 5, 6, 8 |
| `models/shd/local_K8_certified.pt` | locally connected, trained against a K=8 certificate | table 6 |
| `models/shd/dense_K4_control.pt`, `dense_K4_certified.pt` | dense, H=512, D={2,4,8} | tables 2, 6 |
| `models/shd/dense_sweep_lambda*.pt` | penalty-weight sweep from the dense control | table 2 |
| `models/host_dcls/{control,certified}_s{1,2,3}.pt` | recurrent layer inside the learnable-delay (DCLS) network, 3 seeds | table 4 |
| `models/host_spikcommander/{control,certified}_s{1,2,3}.pt` | recurrent layer inside SpikCommander, 3 seeds | table 4 |

Networks of `models/shd/` are evaluated with the training script and zero epochs; the configuration is that of the
run driver that trained them (e.g. `harness/run_speed005_train.sh`). For the certified local network:

```bash
cd harness
DELAYS=2,4,8 AUG=2 SEED=1 CPC=32 LOCAL=1 LOCAL_R=1 H=512 EPOCHS=0 TEST=1 TAG=eval \
  INIT=../models/shd/local_K4_certified.pt python s5_delays.py
```

This reproduces validation accuracy 84.69% and prints test accuracy and certified silence. The host networks are
evaluated inside the reference implementations prepared by `make_ref_protocol.py` and `make_scmd_protocol.py`
(`recert_dcls.py`, `recert_scmd.py`).

## Where each result comes from

| paper item | result files | scripts |
|---|---|---|
| Table 1, figure 2a,b (delay sets, learnable delays) | `results/d1_u1_*`, `d1_D2_*`, `d1_m4_*`, `d1_E3_*`, `d1_t3c_*`, `d1_F2_*`, `d1_t3k_*`, `dc_b_fix_*` | `s5_delays.py`, `s7_dcls.py`, `run_delay00{1,2,3}.sh`, `run_govern002*.sh`, `run_dcls002*.sh`, `rebuild_table1.py` |
| Horizon and core-size re-certification | `results/d1_f_K*_C*` | `run_recertify.sh` |
| Table 2 (penalty sweep) | `results/d1_ar_A0_s1.json`, `d1_g3_lam*` | `run_arch001.sh`, `run_govern003.sh` |
| Table 3, figure 2c (SHD, SSC, N-MNIST) | `results/d1_c_*`, `d1_S1_*`, `d1_S2_*`, `d1_ssc_*`, `d1_ctrl_*`, `d1_cert_*` | `run_shd_confirm_frozen.sh`, `run_ssc_*.sh`, `get_ssc.sh`, `run_nmnist001.sh`, `make_nmnist_h5.py` |
| Table 4 (published architectures) | `results/acc007/`, `a12c_*`, `a11_*`, `acc007_status.txt`, `acc012c_status.txt`, `records/host_dcls_reference.md` | `make_ref_protocol.py`, `acc005_*.py`, `recert_dcls.py`, `make_scmd_protocol.py`, `scmd_*.py`, `recert_scmd.py`, `run_acc0{04,07,11,12b,12c}.sh` |
| 23-network rank correlation (supplement table) | `results/relation.json`, `relation_families.json` | `relation.py`, `relation_families.py` |
| Redistribution across delays, figure 2d,e | `results/d1_E3_*`, `d1_F2_*`, `dc_b_fix_*`, `analysis_ei_ablation.json` | `analyze_ei_ablation.py`, `make_figures.py` |
| Table 5 (bounded fan-in) | `results/d1_dc2048_s1.json`, `d1_lc*`, `d1_lk*` | `run_scalelocal001.sh` |
| Execution, figure 2f, table 6, supplement | `results/core_scaling_002/`, `speed005_probe/`, `speed006_probe/`, `silent_runs.txt`, `records/execution_*.md` | `s6_export_delays.py`, `s6_engine_delays.cpp`, `s8_engine_net.cpp`, `run_manycore.sh`, `run_speed00{5,6}_probe.sh`, `silent_runs.py`, `sm_*.py` |
| Table 7, figures 3 and 4 (hardware) | `results/hw001/`, `hw008/`, `records/hardware.md`, `harness/hw/build/pnr_*/` | `harness/hw/*.py`, `harness/hw/run_*.sh`, `research/paper/figs/fig4_stage.tex` |
| Table 8 (SpiNNaker) | `harness/spinnaker/spinnaker_{pack,result,weights}.npz` | `harness/spinnaker/*.py`, `run_on_spinnaker.ipynb`, `read_weights_spinnaker.ipynb` |

## Requirements

- Python 3.11 with PyTorch, NumPy, h5py and matplotlib; `tonic` for N-MNIST, `spikingjelly` for the SpikCommander host.
- A C++20 compiler with pthreads for the execution engines.
- Hardware flow: Amaranth 0.5, Yosys, OpenROAD, Icarus Verilog, KLayout and the SkyWater 130 nm PDK (`sky130A`).
- SpiNNaker: an EBRAINS account; the notebooks run in the EBRAINS lab with the `EBRAINS-25.10` kernel (sPyNNaker 7.4).
- SageMaker launchers (`harness/sm_*.py`) run in AWS CloudShell.

Scripts download the Heidelberg Spiking Digits on first use into `~/research/data` and keep logs under
`~/research/logs`; checkpoints written by training go to `~/research/models`. The reference implementations of the
learnable-delay network and SpikCommander are cloned under `~/research/refs`. Shell drivers locate the repository
through `git rev-parse` or the `REPO` environment variable.

## Data

The Heidelberg Spiking Digits, Spiking Speech Commands and N-MNIST datasets are publicly available from their authors.
