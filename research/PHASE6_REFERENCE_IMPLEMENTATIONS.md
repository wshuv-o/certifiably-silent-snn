# Phase 6 — Reference Implementations — v1, 2026-10-04

Goal: our DS / SS / AE kernels must not be strawmen. For each mode, there is an established implementation that (a) sets the performance bar our own kernels must reach on identical inputs, and (b) gives an independent measurement of the same comparison.

## 1. Selected references

| Mode / role | Reference | Platform | Why | Source |
|---|---|---|---|---|
| SS (+ dense switching) — CPU bar | **GAP Benchmark Suite (gapbs)** — Beamer et al. | C++/OpenMP, Linux `make` | Reference-quality BFS (direction-optimizing), SSSP (delta-stepping), PageRank, CC, with built-in Kronecker/uniform generators | [sbeamer/gapbs](https://github.com/sbeamer/gapbs), [arXiv:1508.03619](https://arxiv.org/abs/1508.03619) |
| SS with sparse/dense switching — prior-art baseline | **Ligra** — Shun & Blelloch | C++ (Cilk/OpenMP), Linux | The "adaptive switching already exists" threat; must appear as a baseline | Ligra repo by jshun (URL to confirm at clone time) |
| AE — CPU | **Galois** — Pingali group | C++, Linux (CMake) | Asynchronous worklist execution, both optimistic and round-based: a ready-made SS-vs-AE pair on CPU | [IntelligentSoftwareSystems/Galois](https://github.com/IntelligentSoftwareSystems/Galois) |
| AE — GPU | **Atos** — Chen et al., ICPP 2022 | CUDA, Linux | Persistent-kernel async task scheduler for graph analytics, compared against Gunrock (BSP); a ready-made SS-vs-AE pair on GPU | [owensgroup/ATOS](https://github.com/owensgroup/ATOS), [paper](https://par.nsf.gov/servlets/purl/10389255) |
| AE — GPU (alternative) | **Groute** — Ben-Nun et al. | CUDA, Linux | Lock-free GPU worklists for async irregular computation | [groute/groute](https://www.github.com/groute/groute), [TOPC 2020](https://dl.acm.org/doi/fullHtml/10.1145/3399730) |
| SS — GPU | **Gunrock** (Owens group) | CUDA, Linux | Standard BSP frontier framework; Atos's baseline | (URL to confirm) |
| SNN clock-driven — GPU | **Spice** — Kasap & van Opstal | CUDA multi-GPU | Clock-driven SNN reference | [denniskb/spice](https://github.com/denniskb/spice) |
| SNN — CPU/GPU | GeNN, Brian2 (**U**: not re-checked this session) | Python + code generation | Widely used; Brian2 supports both event-driven and clock-driven elements | to confirm |

## 2. Our own harness (what we write)

The references cover graph and SNN workloads, but not our open-loop synthetic generator or a single three-mode comparison. So we write one harness with all three modes, and validate each against the references:

- **CPU:** Numba (≥ 0.63 supports Python 3.14 — verified in the [Numba 0.63 release notes](https://numba.readthedocs.io/en/stable/release/0.63.0-notes.html)).
- **GPU:** CuPy `RawKernel` (CUDA C). CuPy wheels for Python 3.14 are **unverified**; fall back to a Python 3.12 venv if they are missing.
- **Strawman guard (acceptance criterion):** on the same graphs, our SS BFS/PR must come within a stated factor of gapbs (CPU) and Gunrock (GPU), and our AE within a stated factor of Galois/Atos. The factor will be fixed before running; propose ≤ 1.5×. If we miss it, we either fix our kernels or report the reference numbers alongside.
- **Independent check of predictions:** the Galois (round-based vs async) and Gunrock-vs-Atos comparisons provide SS-vs-AE measurements we did not implement, so P2 and P3 can be tested on code we didn't write.

## 3. Graph inputs (span the dimensions)

| Input | Expected position | Source |
|---|---|---|
| Kronecker / RMAT (gapbs `-g`) | high F, low diameter (small D), skewed load (high κ) | gapbs generator |
| Uniform random (gapbs `-u`) | moderate F, low κ | gapbs generator |
| Road networks (e.g., USA-road, SNAP roadNet) | low F, very high diameter (large D) — async expected to struggle (P3) | SNAP / DIMACS |
| Social / web graphs (SNAP) | high F, skewed | SNAP |
| Our open-loop synthetic | factorial control | own generator |

## 4. Platform blocker

Every C++/CUDA reference above builds on Linux. This machine has WSL2 enabled but **no Linux distribution installed** (`WSL_E_DEFAULT_DISTRO_NOT_FOUND`). Recommended: install Ubuntu under WSL2 (`wsl --install -d Ubuntu-24.04`), which also supports CUDA on the RTX 2060 through the existing Windows driver (565.90). Caveat: running under WSL adds a virtualization layer, so all timed runs, including our own harness, should be done in the same environment for comparability.

Risks:
- Atos/Groute may pin older CUDA or GCC versions; RTX 2060 is sm_75.
- Galois needs Boost and LLVM.
- Building may take some iteration. If Atos fails to build, Groute is the fallback for GPU AE.

## Environment log

- 2026-10-04: WSL2 Ubuntu 24.04.5 installed at `D:\WSL\Ubuntu` (user `shuvo`, 12 threads, ~11 GB RAM visible, RTX 2060 visible via driver 565.90). gcc 13.3, cmake 3.28.
- gapbs commit `2972aeb` built in `~/research/refs/gapbs`; smoke test `bfs -g 20 -n 3` → 0.0366 s average (not a measurement, just a build check).
- CUDA 12.6 nvcc (subset: nvcc, cudart, cccl, nvrtc, cuSPARSE, cuBLAS) via NVIDIA's wsl-ubuntu repo. LLVM 18, Boost and fmt for Galois (also needs `libzstd-dev` — LLVM 18 cmake bug).
- Python venv `~/research/venv`: Python 3.12.3, numpy 2.5.3, numba 0.68.0 (12 threads OK), cupy 14.2.0 (RawKernel on the RTX 2060 OK), pynvml, pandas, matplotlib.
- Galois commit `b67f942` (Aug 2023; repo is stale but functional) in `~/research/refs/Galois/build`. Built: bfs-cpu, sssp-cpu, pagerank-push/pull-cpu, graph-convert.
  - **Key finding:** `bfs-cpu --algo=Sync|Async|SyncTile|AsyncTile` and `sssp-cpu --algo=deltaStep|deltaStepBarrier` give **SS vs AE in the same codebase**, with only the execution mode changing. This is the cleanest independent test of P2/P3 available.
- WSL disk use after all installs: 7.5 GB.
- Pending: Gunrock + Atos (GPU SS vs AE), Ligra.

## Next step

Install WSL Ubuntu (needs your approval: system change, ~several GB) → build gapbs first (the simplest), then Galois, Gunrock/Atos → set up the Numba/CuPy harness skeleton. That is the start of Phase 7 (simplest serious baseline).
