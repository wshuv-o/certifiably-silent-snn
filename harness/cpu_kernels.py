"""CPU (Numba) implementations of one execution step.

Step semantics (both modes): every active unit sends its F weighted messages;
each receiving unit accumulates them (state += sum) and becomes active next step
if state > theta. No leak term, so untouched units are unchanged and the two
modes compute identical results.

DS (dense-sync): densify the event list, pull over ALL N*F edges, update ALL units.
SS (sparse-sync): push only from active units. Messages are routed into buckets
owned by destination blocks (one per thread), so updates need no atomics. Each
owner then applies its messages and builds its part of the next frontier.
"""
import numpy as np
from numba import njit, prange, get_num_threads


@njit(parallel=True, cache=False)
def compact(flag):
    N = flag.size
    P = get_num_threads()
    chunk = (N + P - 1) // P
    cnt = np.zeros(P, np.int64)
    for t in prange(P):
        c = 0
        for v in range(t * chunk, min(N, (t + 1) * chunk)):
            c += flag[v]
        cnt[t] = c
    off = np.zeros(P + 1, np.int64)
    for t in range(P):
        off[t + 1] = off[t] + cnt[t]
    out = np.empty(off[P], np.int32)
    for t in prange(P):
        p = off[t]
        for v in range(t * chunk, min(N, (t + 1) * chunk)):
            if flag[v]:
                out[p] = v
                p += 1
    return out


@njit(parallel=True, cache=False)
def ds_step(active, in_ptr, in_idx, in_w, state, theta, x, flag):
    N = state.size
    for v in prange(N):
        x[v] = 0.0
    for i in prange(active.size):
        x[active[i]] = 1.0
    for v in prange(N):
        acc = np.float32(0.0)
        for e in range(in_ptr[v], in_ptr[v + 1]):
            acc += in_w[e] * x[in_idx[e]]
        if acc > 0.0:
            state[v] += acc
            flag[v] = 1 if state[v] > theta else 0
        else:
            flag[v] = 0
    return compact(flag)


@njit(parallel=True, cache=False)
def ss_step(active, dst, w, F, state, theta, touched, tl, bv, bw):
    N = state.size
    P = get_num_threads()
    m = active.size
    blk = (N + P - 1) // P          # destination block owned by each thread
    sch = (m + P - 1) // P          # sender chunk per thread

    # pass 1: count messages per (sender chunk, owner)
    cnt = np.zeros((P, P), np.int64)
    for t in prange(P):
        for i in range(t * sch, min(m, (t + 1) * sch)):
            base = np.int64(active[i]) * F
            for k in range(F):
                cnt[t, dst[base + k] // blk] += 1

    # bucket offsets, grouped by owner
    off = np.zeros((P, P), np.int64)
    ostart = np.zeros(P + 1, np.int64)
    tot = 0
    for o in range(P):
        ostart[o] = tot
        for t in range(P):
            off[t, o] = tot
            tot += cnt[t, o]
    ostart[P] = tot

    # pass 2: route messages into owner buckets
    for t in prange(P):
        pos = off[t].copy()
        for i in range(t * sch, min(m, (t + 1) * sch)):
            base = np.int64(active[i]) * F
            for k in range(F):
                v = dst[base + k]
                o = v // blk
                bv[pos[o]] = v
                bw[pos[o]] = w[base + k]
                pos[o] += 1

    # pass 3: owners apply messages, then build their part of the next frontier
    ncnt = np.zeros(P, np.int64)
    for o in prange(P):
        lo = o * blk
        nt = 0
        for j in range(ostart[o], ostart[o + 1]):
            v = bv[j]
            state[v] += bw[j]
            if touched[v] == 0:
                touched[v] = 1
                tl[lo + nt] = v
                nt += 1
        c = 0
        for j in range(nt):
            v = tl[lo + j]
            touched[v] = 0
            if state[v] > theta:
                tl[lo + c] = v
                c += 1
        ncnt[o] = c

    noff = np.zeros(P + 1, np.int64)
    for o in range(P):
        noff[o + 1] = noff[o] + ncnt[o]
    out = np.empty(noff[P], np.int32)
    for o in prange(P):
        lo = o * blk
        for j in range(ncnt[o]):
            out[noff[o] + j] = tl[lo + j]
    return out


class CPPEngine:
    """ctypes wrapper around cpu_ref.cpp (gapbs-style C++/OpenMP kernels)."""
    LIB = "~/research/build/libcpuref.so"

    def __init__(self, N, F, dst, w, in_ptr, in_idx, in_w):
        import ctypes, os
        self.lib = ctypes.CDLL(os.path.expanduser(self.LIB))
        P, I64, F32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float
        self.lib.ds_step.argtypes = [I64, P, P, P, P, I64, P, F32, P, P]
        self.lib.ss_step.argtypes = [I64, P, P, I64, P, I64, P, F32, P, P, P]
        self.lib.ds_step.restype = self.lib.ss_step.restype = I64
        self.N, self.F = N, F
        self.dst, self.w = np.ascontiguousarray(dst), np.ascontiguousarray(w)
        self.in_ptr, self.in_idx, self.in_w = in_ptr, in_idx, in_w
        self.x = np.zeros(N, np.float32)
        self.touched = np.zeros(N, np.int32)
        self.tl = np.empty(N, np.int32)
        self.next = np.empty(N, np.int32)

    @staticmethod
    def _p(a):
        return a.ctypes.data

    def ds(self, active, state, theta):
        n = self.lib.ds_step(self.N, self._p(self.in_ptr), self._p(self.in_idx), self._p(self.in_w),
                             self._p(active), active.size, self._p(state), theta,
                             self._p(self.x), self._p(self.next))
        return self.next[:n]

    def ss(self, active, state, theta):
        n = self.lib.ss_step(self.N, self._p(self.dst), self._p(self.w), self.F,
                             self._p(active), active.size, self._p(state), theta,
                             self._p(self.touched), self._p(self.tl), self._p(self.next))
        return self.next[:n]


class CPUEngine:
    def __init__(self, N, F, dst, w, in_ptr, in_idx, in_w):
        self.N, self.F = N, F
        self.dst, self.w = dst, w
        self.in_ptr, self.in_idx, self.in_w = in_ptr, in_idx, in_w
        P = get_num_threads()
        blk = (N + P - 1) // P
        self.x = np.zeros(N, np.float32)
        self.flag = np.zeros(N, np.int8)
        self.touched = np.zeros(N, np.int8)
        self.tl = np.empty(P * blk, np.int32)
        self.bv = np.empty(N * F, np.int32)     # preallocated message queues
        self.bw = np.empty(N * F, np.float32)

    def ds(self, active, state, theta):
        return ds_step(active, self.in_ptr, self.in_idx, self.in_w, state,
                       np.float32(theta), self.x, self.flag)

    def ss(self, active, state, theta):
        return ss_step(active, self.dst, self.w, np.int64(self.F), state,
                       np.float32(theta), self.touched, self.tl, self.bv, self.bw)
