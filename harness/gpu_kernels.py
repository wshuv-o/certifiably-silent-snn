"""GPU (CuPy) implementations of one execution step; same semantics as cpu_kernels.

DS: densify the event list, then cuSPARSE SpMV over all N*F edges (a strong dense
    baseline, not hand-written), then update all units.
SS: one thread per active edge pushes with atomicAdd, records newly touched
    units in a list, then a second kernel builds the next frontier from that list.
    The second kernel's grid is sized by an upper bound, so there is no host sync
    between the two kernels.
"""
import cupy as cp
import cupyx.scipy.sparse as cps

_SRC = r'''
extern "C" {
__global__ void densify(const int* active, long long m, float* x) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) x[active[i]] = 1.0f;
}
__global__ void ds_update(const float* y, float* state, float theta, long long N,
                          int* next, unsigned long long* nnext) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) {
        float a = y[v];
        if (a > 0.0f) {
            float s = state[v] + a;
            state[v] = s;
            if (s > theta) next[atomicAdd(nnext, 1ULL)] = (int)v;
        }
    }
}
__global__ void ss_push(const int* active, long long m, int F, const int* dst,
                        const float* w, float* state, int* touched, int* tl,
                        unsigned long long* ntl) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e < m * F) {
        long long i = e / F;
        long long base = (long long)active[i] * F + (e - i * F);
        int v = dst[base];
        atomicAdd(&state[v], w[base]);
        if (atomicExch(&touched[v], 1) == 0) tl[atomicAdd(ntl, 1ULL)] = v;
    }
}
__global__ void ss_frontier(const int* tl, const unsigned long long* ntl, long long bound,
                            int* touched, const float* state, float theta,
                            int* next, unsigned long long* nnext) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < bound && j < (long long)(*ntl)) {
        int v = tl[j];
        touched[v] = 0;
        if (state[v] > theta) next[atomicAdd(nnext, 1ULL)] = v;
    }
}
}
'''
_mod = cp.RawModule(code=_SRC)
_densify = _mod.get_function("densify")
_ds_update = _mod.get_function("ds_update")
_ss_push = _mod.get_function("ss_push")
_ss_frontier = _mod.get_function("ss_frontier")
TPB = 256


def _grid(n):
    return (int((n + TPB - 1) // TPB),)


class GPUEngine:
    def __init__(self, N, F, dst, w, in_ptr, in_idx, in_w):
        self.N, self.F = N, F
        self.dst = cp.asarray(dst)
        self.w = cp.asarray(w)
        self.M = cps.csr_matrix((cp.asarray(in_w), cp.asarray(in_idx), cp.asarray(in_ptr)),
                                shape=(N, N))
        self.x = cp.zeros(N, cp.float32)
        self.touched = cp.zeros(N, cp.int32)
        self.tl = cp.empty(N, cp.int32)
        self.next = cp.empty(N, cp.int32)
        self.cnt = cp.zeros(2, cp.uint64)   # [ntl, nnext]

    def ds(self, active, state, theta):
        m = active.size
        self.cnt.fill(0)
        self.x.fill(0)
        _densify(_grid(m), (TPB,), (active, cp.int64(m), self.x))
        y = self.M.dot(self.x)
        _ds_update(_grid(self.N), (TPB,), (y, state, cp.float32(theta), cp.int64(self.N),
                                           self.next, self.cnt[1:]))
        return self.next, self.cnt

    def ss(self, active, state, theta):
        m = active.size
        self.cnt.fill(0)
        _ss_push(_grid(m * self.F), (TPB,), (active, cp.int64(m), cp.int32(self.F), self.dst,
                                             self.w, state, self.touched, self.tl, self.cnt[:1]))
        bound = min(m * self.F, self.N)
        _ss_frontier(_grid(bound), (TPB,), (self.tl, self.cnt[:1], cp.int64(bound), self.touched,
                                            state, cp.float32(theta), self.next, self.cnt[1:]))
        return self.next, self.cnt

    @staticmethod
    def result(out):
        nxt, cnt = out
        return cp.asnumpy(nxt[: int(cnt[1].get())])
