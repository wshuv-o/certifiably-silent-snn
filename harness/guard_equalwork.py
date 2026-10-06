"""Strawman guard, equal-work check: run our C++ kernels doing exactly the per-edge
work gapbs does (DS without weights ~ PageRank pull; SS without accumulation ~
BFS top-down), on the same graph, and compare ns/edge with the gapbs numbers
from guard_gapbs.py.
"""
import ctypes, os, time
import numpy as np
from graphgen import make_graph, sample_active
from cpu_kernels import CPPEngine

N, F = 1 << 20, 32
dst, w, in_ptr, in_idx, in_w = make_graph(N, F, N, seed=0)
e = CPPEngine(N, F, dst, w, in_ptr, in_idx, in_w)
state0 = np.random.default_rng(1).uniform(0, 1, N).astype(np.float32)
NULL = None

def t(fn, act, reps=15):
    st = state0.copy(); ts = []
    for r in range(reps + 2):
        np.copyto(st, state0); t0 = time.perf_counter(); fn(act, st); ts.append(time.perf_counter() - t0)
    return np.median(ts[2:])

def ds_unweighted(act, st):
    e.lib.ds_step(N, e._p(e.in_ptr), e._p(e.in_idx), NULL, e._p(act), act.size, e._p(st),
                  ctypes.c_float(1.0), e._p(e.x), e._p(e.next))

def ss_noacc(act, st):
    e.lib.ss_step(N, e._p(e.dst), NULL, F, e._p(act), act.size, e._p(st),
                  ctypes.c_float(1.0), e._p(e.touched), e._p(e.tl), e._p(e.next))

rng = np.random.default_rng(2)
print(f"equal-work DS (no weights) ns/edge: {t(ds_unweighted, sample_active(N, 1.0, rng)) / (N*F) * 1e9:.3f}")
for s in (0.001, 0.03, 0.1):
    a = sample_active(N, s, rng)
    print(f"equal-work SS (no accumulate) s={s:<5} ns/edge: {t(ss_noacc, a) / (a.size*F) * 1e9:.3f}")
