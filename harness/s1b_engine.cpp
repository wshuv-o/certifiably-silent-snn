// S1b (pre-registered in research/N3_SCALEUP_PLAN.md): compiled multi-thread "neuromorphic core"
// engine for a ring-local RSNN. Modes: handshake (wait for neighbours' step-t spikes) and cert
// (a neighbour's certificate "silent through step u" also satisfies the wait). O(1)-per-neuron
// certificate from the Proposition: with V_i < theta, neuron i stays silent at every future step tau
// whose input satisfies I_i[tau] <= (1-beta)*theta - R_i  (u <- beta*u + I + R < theta by induction).
// Emulated interconnect latency L: published data is visible to neighbours L ns after publication.
// Spikes must be bit-identical to a single-thread reference.
// Build: g++ -O3 -march=native -std=c++20 -pthread s1b_engine.cpp -o s1b_engine
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
#include <algorithm>

static int P, C, T, N; static double BETA, THETA;
static std::vector<double> Wloc, Iall;            // [P][C][3C], [N][T][H]
static std::vector<double> Rloc;                  // [P][C]
static std::vector<int16_t> NEX;
static int CERT_KIND = 0;                        // 0 = S1b O(1) rule, 1 = S1c Lemma-1 K-step                  // [N][T+2][H]: first tau>=t with I > budget (T if none)

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class X> static void readbin(const std::string& f, std::vector<X>& v, size_t n) {
  v.resize(n); std::ifstream in(f, std::ios::binary); in.read((char*)v.data(), n * sizeof(X));
  if ((size_t)in.gcount() != n * sizeof(X)) { fprintf(stderr, "short read %s\n", f.c_str()); exit(1); }
}

// one core update; identical in engine and reference
static inline void core_step(int p, double* v, const uint8_t* sin, const double* Iext, uint8_t* sout) {
  const double* W = &Wloc[(size_t)p * C * 3 * C];
  for (int i = 0; i < C; ++i) {
    double acc = 0.0;
    for (int j = 0; j < 3 * C; ++j) if (sin[j]) acc += W[i * 3 * C + j];
    double u = BETA * v[i] + Iext[i] + acc;
    uint8_t s = u >= THETA;
    v[i] = u - (s ? THETA : 0.0); sout[i] = s;
  }
}

struct Shared {
  std::vector<uint8_t> spk;                       // [N][T+1][P][C]
  std::unique_ptr<std::atomic<int64_t>[]> pub;    // [N][P][T+1] publish time, 0 = not yet
  std::vector<int32_t> cert;                      // [N][P][T+1] silent through this step
  std::vector<int64_t> tstart, tend;              // [N][P]
};

static void worker(int p, int mode, int64_t L, Shared& S, std::barrier<>& bar) {
  const int H = P * C, left = (p + P - 1) % P, right = (p + 1) % P, nb[3] = {left, p, right};
  std::vector<double> v(C); std::vector<uint8_t> sin(3 * C), s(C);
  auto PUB = [&](int b, int q, int k) -> std::atomic<int64_t>& { return S.pub[((size_t)b * P + q) * (T + 1) + k]; };
  for (int b = 0; b < N; ++b) {
    bar.arrive_and_wait();
    S.tstart[b * P + p] = now_ns();
    std::fill(v.begin(), v.end(), 0.0);
    int last_c = 0, seen[2] = {0, 0};
    const double* I = &Iall[(size_t)b * T * H];
    for (int t = 0; t < T - 1; ++t) {
      for (int qi = 0; qi < 2; ++qi) {
        int q = qi ? right : left;
        for (;;) {
          int k = seen[qi]; int64_t tn = now_ns();
          while (k < T) {
            int64_t ts = PUB(b, q, k + 1).load(std::memory_order_acquire);
            if (ts && tn >= ts + L) ++k; else break;
          }
          seen[qi] = k;
          if (k >= t) break;
          if (mode == 1 && S.cert[((size_t)b * P + q) * (T + 1) + k] >= t) break;
        }
      }
      for (int n = 0; n < 3; ++n)
        std::memcpy(&sin[n * C], &S.spk[(((size_t)b * (T + 1) + t) * P + nb[n]) * C], C);
      core_step(p, v.data(), sin.data(), &I[(size_t)(t + 1) * H + p * C], s.data());
      std::memcpy(&S.spk[(((size_t)b * (T + 1) + t + 1) * P + p) * C], s.data(), C);
      if (mode == 1) {
        bool spiked = std::any_of(s.begin(), s.end(), [](uint8_t x) { return x; });
        if (spiked || t + 1 >= last_c) {
          int c;
          if (CERT_KIND == 0) {                    // S1b: O(1)-per-neuron "unbounded horizon" rule
            c = T;
            for (int i = 0; i < C && c > t + 1; ++i) {
              if (v[i] >= THETA) { c = t + 1; break; }
              int nx = (t + 2 <= T) ? NEX[((size_t)b * (T + 2) + t + 2) * H + p * C + i] : T;
              c = std::min(c, nx - 1);
            }
          } else {                                 // S1c: Lemma 1 K-step bound, KMAX = 16
            int h = 16;
            for (int i = 0; i < C && h > 0; ++i) {
              double u = v[i];
              for (int k = 1; k <= h; ++k) {
                int tau = t + 1 + k;
                if (tau > T - 1) { h = std::min(h, k - 1); break; }
                u = BETA * u + I[(size_t)tau * H + p * C + i] + Rloc[p * C + i];
                if (u >= THETA) { h = k - 1; break; }
              }
            }
            c = t + 1 + h;
          }
          last_c = std::max(c, t + 1);
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
  if (argc > 3) CERT_KIND = atoi(argv[3]);
  { std::ifstream d(dir + "/dims.txt"); d >> P >> C >> T >> N >> BETA >> THETA; }
  const int H = P * C;
  readbin(dir + "/" + tag + "_Wloc.bin", Wloc, (size_t)P * C * 3 * C);
  readbin(dir + "/" + tag + "_I.bin", Iall, (size_t)N * T * H);
  Rloc.assign((size_t)P * C, 0.0);
  for (int p = 0; p < P; ++p) for (int i = 0; i < C; ++i) for (int j = 0; j < 3 * C; ++j)
    Rloc[p * C + i] += std::max(0.0, Wloc[((size_t)p * C + i) * 3 * C + j]);
  NEX.assign((size_t)N * (T + 2) * H, (int16_t)T);  // next exceedance of budget (1-beta)theta - R
  for (int b = 0; b < N; ++b) for (int h = 0; h < H; ++h) {
    double budget = (1.0 - BETA) * THETA - Rloc[h];
    int16_t nx = T;
    for (int tau = T - 1; tau >= 0; --tau) {
      if (Iall[((size_t)b * T + tau) * H + h] > budget) nx = tau;
      NEX[((size_t)b * (T + 2) + tau) * H + h] = nx;
    }
  }
  // reference
  std::vector<uint8_t> ref((size_t)N * (T + 1) * P * C, 0);
  for (int b = 0; b < N; ++b) {
    std::vector<double> v((size_t)P * C, 0.0); std::vector<uint8_t> sin(3 * C);
    for (int t = 0; t < T - 1; ++t) for (int p = 0; p < P; ++p) {
      int nb[3] = {(p + P - 1) % P, p, (p + 1) % P};
      for (int n = 0; n < 3; ++n) std::memcpy(&sin[n * C], &ref[(((size_t)b * (T + 1) + t) * P + nb[n]) * C], C);
      core_step(p, &v[p * C], sin.data(), &Iall[((size_t)b * T + t + 1) * H + p * C],
                &ref[(((size_t)b * (T + 1) + t + 1) * P + p) * C]);
    }
  }
  double meanR = 0; for (double r : Rloc) meanR += r; meanR /= Rloc.size();
  printf("model=%s N=%d meanR=%.4f budget=(1-beta)theta=%.4f\n", tag.c_str(), N, meanR, (1 - BETA) * THETA);
  const int Ls[5] = {0, 5, 20, 100, 500};
  for (int li = 0; li < 5; ++li) for (int mode = 0; mode < 2; ++mode) {
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
    std::vector<double> srt = ms; std::sort(srt.begin(), srt.end());
    bool exact = S.spk == ref;
    // certified coverage: fraction of core-steps where a valid certificate (beyond current step) exists
    size_t cov = 0, tot = 0;
    if (mode == 1) for (int b = 0; b < N; ++b) for (int p = 0; p < P; ++p) for (int t = 1; t < T; ++t) {
      ++tot; cov += S.cert[((size_t)b * P + p) * (T + 1) + t] > t; }
    printf("RESULT kind=%d model=%s L_us=%d mode=%s median_ms=%.4f exact=%d cert_coverage=%.3f\n", CERT_KIND, tag.c_str(), Ls[li],
           mode ? "cert" : "handshake", srt[N / 2], (int)exact, tot ? (double)cov / tot : 0.0);
    fflush(stdout);
  }
}
