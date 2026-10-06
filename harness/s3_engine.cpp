// SWEEP-001 / B (pre-registered): general exact engine. Derived from s2_engine.cpp. Adds:
//  (1) a core waits only on cores that actually project to it (needed for sink hubs, local masks);
//  (2) mode 2 = distributed recursive certificates: when bounding future input from core q at step tau,
//      q's excitation is dropped if q's visible certificate covers step tau-1 (a proven fact).
// Original header: ENGINE-002: exact multi-thread core engine for the
// DENSE 512-ALIF network (16 cores x 32). Each core needs step-t spikes of ALL other cores, or a published
// certificate "silent through step u" from them. ALIF: a <- rho*a + gamma*s_prev; v <- beta*v + I + W s_prev;
// thr = theta + a; spike if v >= thr; v -= s*thr. Certificates (Lemma 1) use the BASE threshold theta
// (a >= 0, so thr >= theta: sound) and worst-case recurrent drive R_i from the first future step.
// Build: g++ -O3 -march=native -std=c++20 -pthread s2_engine.cpp -o s2_engine
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static int P, C, T, N, H; static double BETA, THETA, RHO, GAMMA;
static std::vector<double> W, Iall, R;   // [H][H] (i <- j), [N][T][H], [H]
static std::vector<double> Rq;           // [H][P]: positive drive into neuron i from core q
static std::vector<uint8_t> conn;        // [P][P]: conn[p*P+q] = core q projects to core p

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class X> static void readbin(const std::string& f, std::vector<X>& v, size_t n) {
  v.resize(n); std::ifstream in(f, std::ios::binary); in.read((char*)v.data(), n * sizeof(X));
  if ((size_t)in.gcount() != n * sizeof(X)) { fprintf(stderr, "short read %s\n", f.c_str()); exit(1); }
}

// one core update (neurons p*C .. p*C+C-1); sin = all H spikes of the previous step
static inline void core_step(int p, double* v, double* a, const uint8_t* sin, const double* Iext, uint8_t* sout) {
  for (int ii = 0; ii < C; ++ii) {
    int i = p * C + ii;
    double acc = 0.0;
    const double* Wi = &W[(size_t)i * H];
    for (int j = 0; j < H; ++j) if (sin[j]) acc += Wi[j];
    a[ii] = RHO * a[ii] + GAMMA * (double)sin[i];
    double u = BETA * v[ii] + Iext[i] + acc;
    double thr = THETA + a[ii];
    uint8_t s = u >= thr;
    v[ii] = u - (s ? thr : 0.0); sout[ii] = s;
  }
}

struct Shared {
  std::vector<uint8_t> spk;                       // [N][T+1][H]
  std::unique_ptr<std::atomic<int64_t>[]> pub;    // [N][P][T+1]
  std::vector<int32_t> cert;                      // [N][P][T+1]
  std::vector<int64_t> tstart, tend;              // [N][P]
};

static void worker(int p, int mode, int64_t L, Shared& S, std::barrier<>& bar) {
  std::vector<double> v(C), a(C); std::vector<uint8_t> s(C);
  std::vector<int> seen(P);
  auto PUB = [&](int b, int q, int k) -> std::atomic<int64_t>& { return S.pub[((size_t)b * P + q) * (T + 1) + k]; };
  for (int b = 0; b < N; ++b) {
    bar.arrive_and_wait();
    S.tstart[b * P + p] = now_ns();
    std::fill(v.begin(), v.end(), 0.0); std::fill(a.begin(), a.end(), 0.0); std::fill(seen.begin(), seen.end(), 0);
    int last_c = 0;
    const double* I = &Iall[(size_t)b * T * H];
    for (int t = 0; t < T - 1; ++t) {
      for (int q = 0; q < P; ++q) {
        if (q == p || !conn[p * P + q]) continue;
        for (;;) {
          int k = seen[q]; int64_t tn = now_ns();
          while (k < T) {
            int64_t ts = PUB(b, q, k + 1).load(std::memory_order_acquire);
            if (ts && tn >= ts + L) ++k; else break;
          }
          seen[q] = k;
          if (k >= t) break;
          if (mode == 1 && S.cert[((size_t)b * P + q) * (T + 1) + k] >= t) break;
        }
      }
      const uint8_t* sin = &S.spk[((size_t)b * (T + 1) + t) * H];
      core_step(p, v.data(), a.data(), sin, &I[(size_t)(t + 1) * H], s.data());
      std::memcpy(&S.spk[((size_t)b * (T + 1) + t + 1) * H + p * C], s.data(), C);
      if (mode >= 1) {
        bool spiked = std::any_of(s.begin(), s.end(), [](uint8_t x) { return x; });
        if (mode == 2) {                                   // distributed recursive certificate
          // q is provably silent at steps in (seen[q], cert_q]; published steps <= seen[q] may hold real spikes
          std::vector<int> lo(P, 0), hi(P, -1);
          for (int q = 0; q < P; ++q) if (q != p && conn[p * P + q]) {
            lo[q] = seen[q]; hi[q] = S.cert[((size_t)b * P + q) * (T + 1) + seen[q]]; }
          int h = 16;
          for (int ii = 0; ii < C && h > 0; ++ii) {
            int i = p * C + ii; double u = v[ii];
            for (int k = 1; k <= h; ++k) {
              int tau = t + 1 + k;
              if (tau > T - 1) { h = std::min(h, k - 1); break; }
              double Rt = Rq[(size_t)i * P + p];           // own core: worst case
              for (int q = 0; q < P; ++q) if (q != p && conn[p * P + q] && !(tau - 1 > lo[q] && tau - 1 <= hi[q])) Rt += Rq[(size_t)i * P + q];
              u = BETA * u + I[(size_t)tau * H + i] + Rt;
              if (u >= THETA) { h = k - 1; break; }
            }
          }
          last_c = std::max(last_c > t + 1 ? last_c : 0, t + 1 + h);
        } else if (spiked || t + 1 >= last_c) {
          int h = 16;
          for (int ii = 0; ii < C && h > 0; ++ii) {
            int i = p * C + ii; double u = v[ii];
            for (int k = 1; k <= h; ++k) {
              int tau = t + 1 + k;
              if (tau > T - 1) { h = std::min(h, k - 1); break; }
              u = BETA * u + I[(size_t)tau * H + i] + R[i];
              if (u >= THETA) { h = k - 1; break; }
            }
          }
          last_c = (spiked ? t + 1 : t + 1 + h);
          if (spiked) {               // recompute from the post-reset state as well (still sound)
            int h2 = 16;
            for (int ii = 0; ii < C && h2 > 0; ++ii) {
              int i = p * C + ii; double u = v[ii];
              for (int k = 1; k <= h2; ++k) {
                int tau = t + 1 + k;
                if (tau > T - 1) { h2 = std::min(h2, k - 1); break; }
                u = BETA * u + I[(size_t)tau * H + i] + R[i];
                if (u >= THETA) { h2 = k - 1; break; }
              }
            }
            last_c = t + 1 + h2;
          }
        }
        S.cert[((size_t)b * P + p) * (T + 1) + t + 1] = last_c;
      }
      PUB(b, p, t + 1).store(now_ns(), std::memory_order_release);
    }
    S.tend[b * P + p] = now_ns();
  }
}

int main(int argc, char** argv) {
  std::string dir = argv[1], tag = argv[2];
  { std::ifstream d(dir + "/" + tag + "_dims.txt"); d >> P >> C >> T >> N >> BETA >> THETA >> RHO >> GAMMA; }
  H = P * C;
  readbin(dir + "/" + tag + "_W.bin", W, (size_t)H * H);
  readbin(dir + "/" + tag + "_I.bin", Iall, (size_t)N * T * H);
  R.assign(H, 0.0); Rq.assign((size_t)H * P, 0.0); conn.assign((size_t)P * P, 0);
  for (int i = 0; i < H; ++i) for (int j = 0; j < H; ++j) {
    double w = W[(size_t)i * H + j]; R[i] += std::max(0.0, w); Rq[(size_t)i * P + j / C] += std::max(0.0, w);
    if (w != 0.0) conn[(i / C) * P + j / C] = 1; }
  // reference (single thread, same arithmetic)
  std::vector<uint8_t> ref((size_t)N * (T + 1) * H, 0);
  for (int b = 0; b < N; ++b) {
    std::vector<double> v(H, 0.0), a(H, 0.0);
    for (int t = 0; t < T - 1; ++t) for (int p = 0; p < P; ++p)
      core_step(p, &v[p * C], &a[p * C], &ref[((size_t)b * (T + 1) + t) * H], &Iall[((size_t)b * T + t + 1) * H],
                &ref[((size_t)b * (T + 1) + t + 1) * H + p * C]);
  }
  double meanR = 0; for (double r : R) meanR += r; meanR /= H;
  printf("model=%s N=%d cores=%d meanR=%.4f\n", tag.c_str(), N, P, meanR);
  const int Ls[5] = {0, 5, 20, 100, 500};
  int nconn = 0; for (int p = 0; p < P; ++p) for (int q = 0; q < P; ++q) nconn += (p != q) && conn[p * P + q];
  printf("inter-core edges (directed) = %d of %d\n", nconn, P * (P - 1));
  const int MAXMODE = argc > 3 ? atoi(argv[3]) : 3;     // final runs: 2 (handshake, cert)
  for (int li = 0; li < 5; ++li) for (int mode = 0; mode < MAXMODE; ++mode) {
    Shared S; S.spk.assign(ref.size(), 0);
    S.pub.reset(new std::atomic<int64_t>[(size_t)N * P * (T + 1)]);
    for (size_t i = 0; i < (size_t)N * P * (T + 1); ++i) S.pub[i].store(i % (T + 1) == 0 ? 1 : 0);
    S.cert.assign((size_t)N * P * (T + 1), 0); S.tstart.assign(N * P, 0); S.tend.assign(N * P, 0);
    std::barrier bar(P);
    std::vector<std::thread> th;
    for (int p = 0; p < P; ++p) th.emplace_back(worker, p, mode, (int64_t)Ls[li] * 1000, std::ref(S), std::ref(bar));
    for (auto& x : th) x.join();
    std::vector<double> ms(N);
    for (int b = 0; b < N; ++b) {
      int64_t s0 = *std::min_element(&S.tstart[b * P], &S.tstart[b * P] + P);
      int64_t s1 = *std::max_element(&S.tend[b * P], &S.tend[b * P] + P);
      ms[b] = (s1 - s0) / 1e6;
    }
    std::sort(ms.begin(), ms.end());
    size_t cov = 0, tot = 0;
    if (mode >= 1) for (int b = 0; b < N; ++b) for (int p = 0; p < P; ++p) for (int t = 1; t < T; ++t) {
      ++tot; cov += S.cert[((size_t)b * P + p) * (T + 1) + t] > t; }
    printf("RESULT model=%s L_us=%d mode=%s median_ms=%.4f exact=%d cert_coverage=%.3f\n", tag.c_str(), Ls[li],
           mode == 0 ? "handshake" : (mode == 1 ? "cert" : "dcert"), ms[N / 2], (int)(S.spk == ref), tot ? (double)cov / tot : 0.0);
    fflush(stdout);
  }
}
