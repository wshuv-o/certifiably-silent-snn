"""Synthetic workload generator (open-loop mode, Phase 4 §3).

Regular out-degree graph: every unit has exactly F out-edges, so per-event work
is fixed and sparsity is the only thing that changes how much work there is.
Locality is set by a destination window; activity is sampled independently of
the graph (open-loop).
"""
import numpy as np


def make_graph(N, F, window, seed=0):
    """Return (dst, w, in_ptr, in_idx, in_w).

    dst, w : out-edges, unit u owns slots [u*F, (u+1)*F)   (push / event mode)
    in_*   : CSR by destination                             (pull / dense mode)
    window : destinations drawn uniformly from u + [-window/2, window/2] mod N;
             window >= N means uniformly random destinations.
    Weights are strictly positive, so "received input" <=> "accumulated > 0".
    """
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(N, dtype=np.int64), F)
    if window >= N:
        dst = rng.integers(0, N, size=N * F, dtype=np.int64)
    else:
        off = rng.integers(-(window // 2), window // 2 + 1, size=N * F, dtype=np.int64)
        dst = (src + off) % N
    w = rng.uniform(0.01, 1.0, size=N * F).astype(np.float32)

    order = np.argsort(dst, kind="stable")
    in_idx = src[order].astype(np.int32)
    in_w = w[order]
    in_ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(dst, minlength=N), out=in_ptr[1:])
    return dst.astype(np.int32), w, in_ptr, in_idx, in_w


def sample_active(N, s, rng, spread=1.0):
    """Active set of size round(s*N), sorted (as a compacted frontier would be).

    spread = 1   : uniform over all units (spatial imbalance kappa ~ 1)
    spread = rho : drawn uniformly from one random contiguous region of
                   max(m, rho*N) units, which concentrates activity spatially
    """
    m = max(1, int(round(s * N)))
    if spread >= 1.0:
        return np.sort(rng.choice(N, m, replace=False)).astype(np.int32)
    R = max(m, int(round(spread * N)))
    start = int(rng.integers(0, N - R + 1))
    return np.sort(start + rng.choice(R, m, replace=False)).astype(np.int32)


def owner_imbalance(active, dst, F, N, P):
    """kappa = max/mean messages received per destination owner (P contiguous
    blocks, as in the CPU SS kernel)."""
    e = (active.astype(np.int64)[:, None] * F + np.arange(F)).ravel()
    load = np.bincount(dst[e] // ((N + P - 1) // P), minlength=P)
    return float(load.max() / load.mean())


def reference_step(active, dst, w, F, state0, theta):
    """Independent NumPy reference for one step (float64 accumulation)."""
    add = np.zeros(state0.size, np.float64)
    for c in range(0, active.size, 1 << 18):   # chunked to bound memory at large N
        e = (active[c:c + (1 << 18)].astype(np.int64)[:, None] * F + np.arange(F)).ravel()
        add += np.bincount(dst[e], weights=w[e].astype(np.float64), minlength=state0.size)
    state = state0.astype(np.float64) + add
    nxt = np.nonzero((add > 0) & (state > theta))[0]
    return state, nxt
