// C++/OpenMP CPU kernels (gapbs-style), same step semantics as cpu_kernels.py.
//   DS: densify events, pull over all in-edges, update all units, compact next frontier.
//   SS: single-pass push from the active list with atomic float add; newly touched
//       units go into thread-local buffers (as in gapbs QueueBuffer), then a second
//       parallel pass builds the next frontier from the touched list.
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC cpu_ref.cpp -o libcpuref.so
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

static int64_t gather(std::vector<std::vector<int32_t>>& loc, int32_t* out) {
    int T = (int)loc.size();
    std::vector<int64_t> off(T + 1, 0);
    for (int t = 0; t < T; ++t) off[t + 1] = off[t] + (int64_t)loc[t].size();
    #pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < T; ++t)
        if (!loc[t].empty()) std::memcpy(out + off[t], loc[t].data(), loc[t].size() * sizeof(int32_t));
    return off[T];
}

extern "C" {

int64_t ds_step(int64_t N, const int64_t* in_ptr, const int32_t* in_idx, const float* in_w,
                const int32_t* active, int64_t m, float* state, float theta,
                float* x, int32_t* next) {
    #pragma omp parallel for schedule(static)
    for (int64_t v = 0; v < N; ++v) x[v] = 0.0f;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) x[active[i]] = 1.0f;
    int T = omp_get_max_threads();
    std::vector<std::vector<int32_t>> loc(T);
    #pragma omp parallel
    {
        auto& my = loc[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (int64_t v = 0; v < N; ++v) {
            float acc = 0.0f;
            if (in_w) { for (int64_t e = in_ptr[v]; e < in_ptr[v + 1]; ++e) acc += in_w[e] * x[in_idx[e]]; }
            else { for (int64_t e = in_ptr[v]; e < in_ptr[v + 1]; ++e) acc += x[in_idx[e]]; }  // gapbs-PR-equivalent work
            if (acc > 0.0f) {
                float s = state[v] + acc;
                state[v] = s;
                if (s > theta) my.push_back((int32_t)v);
            }
        }
    }
    return gather(loc, next);
}

int64_t ss_step(int64_t N, const int32_t* dst, const float* w, int64_t F,
                const int32_t* active, int64_t m, float* state, float theta,
                int32_t* touched, int32_t* tl, int32_t* next) {
    int T = omp_get_max_threads();
    std::vector<std::vector<int32_t>> loc(T);
    #pragma omp parallel
    {
        auto& my = loc[omp_get_thread_num()];
        #pragma omp for schedule(dynamic, 64)
        for (int64_t i = 0; i < m; ++i) {
            const int64_t base = (int64_t)active[i] * F;
            for (int64_t k = 0; k < F; ++k) {
                const int32_t v = dst[base + k];
                if (w) {  // w == nullptr: BFS-top-down-equivalent work (no accumulation)
                    #pragma omp atomic update
                    state[v] += w[base + k];
                }
                if (__atomic_load_n(&touched[v], __ATOMIC_RELAXED) == 0 &&
                    __atomic_exchange_n(&touched[v], 1, __ATOMIC_RELAXED) == 0)
                    my.push_back(v);
            }
        }
    }
    int64_t nt = gather(loc, tl);
    for (auto& l : loc) l.clear();
    #pragma omp parallel
    {
        auto& my = loc[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (int64_t j = 0; j < nt; ++j) {
            const int32_t v = tl[j];
            touched[v] = 0;
            if (state[v] > theta) my.push_back(v);
        }
    }
    return gather(loc, next);
}

}  // extern "C"

