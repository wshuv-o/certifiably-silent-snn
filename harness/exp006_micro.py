"""EXP-006: workload-independent memory-primitive costs vs working-set size M.

No graph, no frontier: E random indices drawn uniformly from [0, M).
GPU (CuPy):
  gather : y[i] = w[i] * x[idx[i]]                       (ns per element)
  atomic : atomicAdd(&s[idx[i]], w[i]); atomicExch(&f[idx[i]], 1)
  launch : empty-kernel launch overhead                    (us)
CPU (Numba, 6 threads, active wait):
  gather : as above, parallel over i
  stream : copy of 8-byte records (bucket write + read)    (ns per record)
  rmw    : s[v] += w; flag check/set, v uniform within the thread's own block
           of M/P units (as in owner-computes apply)      (ns per element)
"""
import os, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
os.environ.setdefault("OMP_WAIT_POLICY", "active")
import numpy as np
from numba import njit, prange, get_num_threads

E = 1 << 25
LOG2M = [17, 18, 19, 20, 21, 22]
REPS = 9


def med_time(fn, reps=REPS, sync=None):
    ts = []
    for _ in range(reps + 2):
        t0 = time.perf_counter(); fn()
        if sync: sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[2:]))


@njit(parallel=True)
def cpu_gather(idx, w, x, y):
    for i in prange(idx.size):
        y[i] = w[i] * x[idx[i]]


@njit(parallel=True)
def cpu_stream(src_v, src_w, dst_v, dst_w):
    for i in prange(src_v.size):
        dst_v[i] = src_v[i]; dst_w[i] = src_w[i]


@njit(parallel=True)
def cpu_rmw(idx_local, w, s, flag, P, blk):
    n = idx_local.size // P
    for t in prange(P):
        base = t * blk
        for j in range(t * n, (t + 1) * n):
            v = base + idx_local[j]
            s[v] += w[j]
            if flag[v] == 0:
                flag[v] = 1


def cpu_costs(rng):
    out = {}
    w = rng.uniform(0.01, 1, E).astype(np.float32)
    y = np.empty(E, np.float32)
    v2 = np.empty(E, np.int32); w2 = np.empty(E, np.float32)
    P = get_num_threads()
    vsrc = rng.integers(0, 1 << 20, E).astype(np.int32)
    out["stream_ns"] = med_time(lambda: cpu_stream(vsrc, w, v2, w2)) / E * 1e9
    for k in LOG2M:
        M = 1 << k
        idx = rng.integers(0, M, E).astype(np.int32)
        x = rng.uniform(0, 1, M).astype(np.float32)
        g = med_time(lambda: cpu_gather(idx, w, x, y)) / E * 1e9
        blk = (M + P - 1) // P
        il = rng.integers(0, blk, E).astype(np.int32)
        s = np.zeros(P * blk, np.float32); f = np.zeros(P * blk, np.int8)
        def rmw():
            f[:] = 0
            cpu_rmw(il, w, s, f, P, blk)
        r = med_time(rmw) / E * 1e9
        out[k] = {"gather_ns": g, "rmw_ns": r}
        print("cpu", k, out[k], flush=True)
    return out


def gpu_costs(rng):
    import cupy as cp
    src = r'''
    extern "C" {
    __global__ void gather(const int* idx, const float* w, const float* x, float* y, long long E) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < E) y[i] = w[i] * x[idx[i]];
    }
    __global__ void atom(const int* idx, const float* w, float* s, int* f, long long E) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < E) { int v = idx[i]; atomicAdd(&s[v], w[i]); atomicExch(&f[v], 1); }
    }
    __global__ void empty() {}
    }'''
    mod = cp.RawModule(code=src)
    gk, ak, ek = mod.get_function("gather"), mod.get_function("atom"), mod.get_function("empty")
    sync = cp.cuda.Device().synchronize
    grid = ((E + 255) // 256,)
    w = cp.asarray(rng.uniform(0.01, 1, E).astype(np.float32))
    y = cp.empty(E, cp.float32)
    out = {"launch_us": med_time(lambda: ek((1,), (32,), ()), reps=200, sync=sync) * 1e6}
    for k in LOG2M:
        M = 1 << k
        idx = cp.asarray(rng.integers(0, M, E).astype(np.int32))
        x = cp.asarray(rng.uniform(0, 1, M).astype(np.float32))
        s = cp.zeros(M, cp.float32); f = cp.zeros(M, cp.int32)
        # warm clocks, then time
        for _ in range(20): gk(grid, (256,), (idx, w, x, y, cp.int64(E)))
        g = med_time(lambda: gk(grid, (256,), (idx, w, x, y, cp.int64(E))), sync=sync) / E * 1e9
        a = med_time(lambda: ak(grid, (256,), (idx, w, s, f, cp.int64(E))), sync=sync) / E * 1e9
        out[k] = {"gather_ns": g, "atomic_ns": a}
        print("gpu", k, out[k], flush=True)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    res = {"cpu": cpu_costs(rng), "gpu": gpu_costs(rng), "E": E}
    json.dump(res, open("../results/exp006_micro.json", "w"), indent=1)
