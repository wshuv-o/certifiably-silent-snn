// DELAY-ENGINE-001: exact multi-core engine for the MULTI-TAP DELAY network (s5_delays.py).
//
// Derived from s3_engine.cpp, which assumed one-step recurrence. The recurrent drive is now
//   v(t+1) = BETA*v(t) + I(t+1) + sum_d W_d s(t+1-d)
// which changes BOTH the baseline and the certificate, in opposite directions:
//
//  (1) THE BASELINE GETS STRONGER, and the comparison must reflect that or it is a strawman.
//      To compute step t+1 a core needs neighbour spikes only up to step t+1-d_min, not up to t.
//      With d_min = 2 every core is permanently one step ahead with no certificate at all. This is
//      exactly the lookahead conservative PDES takes from minimum synaptic delay, so mode 0
//      ("handshake") is given it. Measuring certificates against a delay-naive handshake would
//      manufacture a speed-up that belongs to the delays.
//
//  (2) THE CERTIFICATE GETS STRONGER. At horizon k the arrival time of tap d is ts = t+1+k-d.
//      For ts <= t+1 those spikes are already determined, so they are used EXACTLY (for the core's
//      own neurons always; for a neighbour q when ts <= seen[q]). Only genuinely future arrivals are
//      bounded by the worst-case per-tap drive Rq_d. This is what lets a trained net certify at all:
//      it concentrates excitation in the long taps (d=4,8), which are exact at short horizons, and
//      starves the short tap (d=2), which is the only one bounded there.
//
// Soundness: every bound replaces an unknown spike pattern with the worst case (all presynaptic
// neurons firing, positive weights only), and the base threshold THETA is used while the true
// threshold is THETA + a with a >= 0. Both are conservative. The engine verifies spike trains
// bit-for-bit against a single-thread reference on every run; exact=0 invalidates all timings.
//
// Build: g++ -O3 -march=native -std=c++20 -pthread s6_engine_delays.cpp -o s6_engine_delays
// Run:   s6_engine_delays <dir> <tag> [maxmode]
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

static int P, C, T, N, H, ND, DMIN;
static double BETA, THETA, RHO, GAMMA;
static std::vector<int> DEL;                       // delay of each tap, ascending
static std::vector<std::vector<double>> Wd;        // [tap][H*H]  (i <- j)
static std::vector<double> Iall;                   // [N][T][H]
static std::vector<std::vector<double>> Rqd;       // [tap][H*P]: positive drive into i from core q
static std::vector<uint8_t> conn;                  // [P][P]: conn[p*P+q] = q projects to p (any tap)

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class X> static void readbin(const std::string& f, std::vector<X>& v, size_t n) {
  v.resize(n); std::ifstream in(f, std::ios::binary); in.read((char*)v.data(), n * sizeof(X));
  if ((size_t)in.gcount() != n * sizeof(X)) { fprintf(stderr, "short read %s\n", f.c_str()); exit(1); }
}

// One core update producing step t+1 for neurons p*C .. p*C+C-1.
// spk is the full [T+1][H] spike array; tap d reads step t+1-d. Adaptation uses step t (as in s5).
static inline void core_step(int p, double* v, double* a, const uint8_t* spk, int t,
                             const double* Iext, uint8_t* sout) {
  for (int ii = 0; ii < C; ++ii) {
    int i = p * C + ii;
    double acc = 0.0;
    for (int di = 0; di < ND; ++di) {
      int ts = t + 1 - DEL[di];
      if (ts < 0) continue;
      const uint8_t* sin = &spk[(size_t)ts * H];
      const double* Wi = &Wd[di][(size_t)i * H];
      for (int j = 0; j < H; ++j) if (sin[j]) acc += Wi[j];
    }
    a[ii] = RHO * a[ii] + GAMMA * (double)spk[(size_t)t * H + i];
    double u = BETA * v[ii] + Iext[i] + acc;
    double thr = THETA + a[ii];
    uint8_t s = u >= thr;
    v[ii] = u - (s ? thr : 0.0); sout[ii] = s;
  }
}

struct Shared {
  std::vector<uint8_t> spk;                       // [N][T+1][H]
  std::unique_ptr<std::atomic<int64_t>[]> pub;    // [N][P][T+1] publication timestamps
  std::vector<int32_t> cert;                      // [N][P][T+1] "silent through step c"
  std::vector<int64_t> tstart, tend;              // [N][P]
  std::vector<int64_t> wait_ns, comp_ns;          // [P] per-core totals, summed over all samples
};

// --- certificate cost optimisation -------------------------------------------------------------
// The exact part of the bound is a sum over FIRING presynaptic neurons, and the firing rate is ~4%,
// so ~20 of 512 neurons fire per step. Walking the full weight row per neuron per horizon (the
// obvious implementation) made the certificate far slower than the handshake it must beat. Instead
// build per-core lists of firing indices once per step, for the steps a certificate can read
// (ts <= t+1, at most DMAX of them), then touch weights only at those positions.
struct FireLists {
  std::vector<std::vector<std::vector<int>>> fire;   // fire[r][q] = firing indices of core q at base+r
  int base = 0, nr = 0;
};

static void build_fires(FireLists& F, const uint8_t* spk, int t) {
  static const int DMAX = *std::max_element(DEL.begin(), DEL.end());
  int lo = std::max(0, t + 2 - DMAX), hi = t + 1;
  F.base = lo; F.nr = hi - lo + 1;
  if ((int)F.fire.size() < F.nr) F.fire.resize(F.nr);
  for (int r = 0; r < F.nr; ++r) {
    if ((int)F.fire[r].size() < P) F.fire[r].resize(P);
    const uint8_t* sin = &spk[(size_t)(lo + r) * H];
    for (int q = 0; q < P; ++q) {
      F.fire[r][q].clear();
      for (int jj = 0; jj < C; ++jj) { int j = q * C + jj; if (sin[j]) F.fire[r][q].push_back(j); }
    }
  }
}

// Worst-case drive into neuron i at horizon k of a certificate rooted after step t+1.
// Exact wherever the arrival time is already determined AND visible; worst-case bounded otherwise.
static inline double drive_bound(int p, int i, int t, int k, const FireLists& F,
                                 const std::vector<int>& seen) {
  double acc = 0.0;
  for (int di = 0; di < ND; ++di) {
    int ts = t + 1 + k - DEL[di];
    if (ts < 0) continue;
    const double* Wi = &Wd[di][(size_t)i * H];
    for (int q = 0; q < P; ++q) {
      if (q != p && !conn[p * P + q]) continue;
      bool known = (ts <= t + 1) && (q == p || ts <= seen[q]);
      if (known) { for (int j : F.fire[ts - F.base][q]) acc += Wi[j]; }
      else       { acc += Rqd[di][(size_t)i * P + q]; }
    }
  }
  return acc;
}

// ACC-012c (CERT_ADAPT=1): adaptation-aware threshold. Before a neuron's first spike inside the window its
// adaptation only decays, so at step t+1+k the threshold is exactly THETA + RHO^k a + GAMMA RHO^(k-1) s, with
// a and s the neuron's adaptation and spike at step t+1. CERT_ADAPT unset: the base threshold THETA.
static bool ADAPT = false;
// SPEED-004 (CERT_FAST=1): the same bound, computed with per-core work hoisted out of the neuron loop.
// Which presynaptic cores are known at (t, k, tap) is the same for every neuron of the core, so the firing
// list of known cores and the list of unknown cores are built once per (k, tap); a tap whose arrival lies
// beyond the certificate root (ts > t+1) is unknown for every core and uses a per-neuron total precomputed
// at load. The certificate compares against THETA - 1e-12 so summation-order rounding can only make the
// bound more conservative; spike trains are still verified bit-for-bit on every run.
static bool FAST = false;
static const uint8_t* REFP = nullptr;      // reference spike trains, for the oracle mode only
static std::vector<std::vector<double>> Rtot;      // [tap][H]: sum over connected q (incl. own core) of Rqd

static int cert_horizon(int p, int t, const double* v, const double* a, const uint8_t* sown, const double* I,
                        const FireLists& F, const std::vector<int>& seen, int hmax) {
  if (!FAST) {
    int h = hmax;
    for (int ii = 0; ii < C && h > 0; ++ii) {
      int i = p * C + ii; double u = v[ii];
      double rk = 1.0;                                   // RHO^k, built incrementally
      for (int k = 1; k <= h; ++k) {
        int tau = t + 1 + k;
        if (tau > T - 1) { h = std::min(h, k - 1); break; }
        u = BETA * u + I[(size_t)tau * H + i] + drive_bound(p, i, t, k, F, seen);
        double thr = THETA;
        if (ADAPT) { thr += rk * RHO * a[ii] + GAMMA * rk * (double)sown[ii]; }
        rk *= RHO;
        if (u >= thr) { h = k - 1; break; }
      }
    }
    return h;
  }
  // ---- fast path: plan every (k, tap) once for the core ----
  struct Plan { int ts; bool beyond; std::vector<int> fires, unk; };
  static thread_local std::vector<Plan> plan;
  const int nk = std::max(0, std::min(hmax, T - 2 - t));
  plan.resize((size_t)(nk + 1) * ND);
  for (int k = 1; k <= nk; ++k) for (int di = 0; di < ND; ++di) {
    Plan& pl = plan[(size_t)k * ND + di];
    pl.ts = t + 1 + k - DEL[di]; pl.fires.clear(); pl.unk.clear(); pl.beyond = pl.ts > t + 1;
    if (pl.ts < 0 || pl.beyond) continue;
    for (int q = 0; q < P; ++q) {
      if (q != p && !conn[p * P + q]) continue;
      if (q == p || pl.ts <= seen[q]) { const auto& fl = F.fire[pl.ts - F.base][q]; pl.fires.insert(pl.fires.end(), fl.begin(), fl.end()); }
      else pl.unk.push_back(q);
    }
  }
  int h = nk;
  for (int ii = 0; ii < C && h > 0; ++ii) {
    int i = p * C + ii; double u = v[ii];
    double rk = 1.0;
    for (int k = 1; k <= h; ++k) {
      int tau = t + 1 + k;
      double acc = 0.0;
      for (int di = 0; di < ND; ++di) {
        const Plan& pl = plan[(size_t)k * ND + di];
        if (pl.ts < 0) continue;
        if (pl.beyond) { acc += Rtot[di][i]; continue; }
        const double* Wi = &Wd[di][(size_t)i * H];
        for (int j : pl.fires) acc += Wi[j];
        const double* Ri = &Rqd[di][(size_t)i * P];
        for (int q : pl.unk) acc += Ri[q];
      }
      u = BETA * u + I[(size_t)tau * H + i] + acc;
      double thr = THETA - 1e-12;
      if (ADAPT) { thr += rk * RHO * a[ii] + GAMMA * rk * (double)sown[ii]; }
      rk *= RHO;
      if (u >= thr) { h = k - 1; break; }
    }
  }
  return h;
}

static void worker(int p, int mode, int64_t L, Shared& S, std::barrier<>& bar) {
  std::vector<double> v(C), a(C); std::vector<uint8_t> s(C);
  std::vector<int> seen(P); FireLists F;
  auto PUB = [&](int b, int q, int k) -> std::atomic<int64_t>& { return S.pub[((size_t)b * P + q) * (T + 1) + k]; };
  int64_t wait_acc = 0, comp_acc = 0;
  for (int b = 0; b < N; ++b) {
    bar.arrive_and_wait();
    S.tstart[b * P + p] = now_ns();
    std::fill(v.begin(), v.end(), 0.0); std::fill(a.begin(), a.end(), 0.0); std::fill(seen.begin(), seen.end(), 0);
    int last_c = 0;
    const double* I = &Iall[(size_t)b * T * H];
    uint8_t* spk = &S.spk[(size_t)b * (T + 1) * H];
    for (int t = 0; t < T - 1; ++t) {
      // To compute step t+1 we need neighbour spikes only up to t+1-DMIN: that is the free lookahead
      // the minimum synaptic delay provides, and mode 0 gets it too.
      const int need = t + 1 - DMIN;
      const int64_t w0 = now_ns();
      for (int q = 0; q < P; ++q) {
        if (q == p || !conn[p * P + q]) continue;
        for (;;) {
          int k = seen[q]; int64_t tn = now_ns();
          while (k < T) {
            int64_t ts = PUB(b, q, k + 1).load(std::memory_order_acquire);
            if (ts && tn >= ts + L) ++k; else break;
          }
          seen[q] = k;
          if (k >= need) break;
          if (mode >= 1 && S.cert[((size_t)b * P + q) * (T + 1) + k] >= need) break;
        }
      }
      const int64_t w1 = now_ns();
      wait_acc += w1 - w0;
      core_step(p, v.data(), a.data(), spk, t, &I[(size_t)(t + 1) * H], s.data());
      std::memcpy(&spk[(size_t)(t + 1) * H + p * C], s.data(), C);
      if (mode == 2) {
        // ORACLE (upper bound, not a runtime method): the core's true silent run, read from the reference
        // spike trains, capped at the same 16-step horizon as the certificate.
        const uint8_t* R = &REFP[(size_t)b * (T + 1) * H];
        int h = 0;
        for (int k = 1; k <= 16 && t + 1 + k <= T - 1; ++k) {
          const uint8_t* row = &R[(size_t)(t + 1 + k) * H + p * C];
          if (std::any_of(row, row + C, [](uint8_t x) { return x; })) break;
          h = k;
        }
        S.cert[((size_t)b * P + p) * (T + 1) + t + 1] = t + 1 + h;
      } else if (mode >= 1) {
        bool spiked = std::any_of(s.begin(), s.end(), [](uint8_t x) { return x; });
        if (spiked || t + 1 >= last_c) {
          build_fires(F, spk, t);
          last_c = t + 1 + cert_horizon(p, t, v.data(), a.data(), s.data(), I, F, seen, 16);
        }
        S.cert[((size_t)b * P + p) * (T + 1) + t + 1] = last_c;
      }
      PUB(b, p, t + 1).store(now_ns(), std::memory_order_release);
      comp_acc += now_ns() - w1;
    }
    S.tend[b * P + p] = now_ns();
  }
  S.wait_ns[p] = wait_acc;
  S.comp_ns[p] = comp_acc;
}

int main(int argc, char** argv) {
  std::string dir = argv[1], tag = argv[2];
  { const char* e = std::getenv("CERT_ADAPT"); ADAPT = e && std::string(e) == "1"; }
  { const char* e = std::getenv("CERT_FAST"); FAST = e && std::string(e) == "1"; }
  {
    std::ifstream d(dir + "/" + tag + "_dims.txt");
    d >> P >> C >> T >> N >> BETA >> THETA >> RHO >> GAMMA >> ND;
    DEL.resize(ND); for (int i = 0; i < ND; ++i) d >> DEL[i];
  }
  H = P * C;
  DMIN = *std::min_element(DEL.begin(), DEL.end());
  Wd.resize(ND); Rqd.assign(ND, {});
  for (int di = 0; di < ND; ++di) readbin(dir + "/" + tag + "_W" + std::to_string(di) + ".bin", Wd[di], (size_t)H * H);
  readbin(dir + "/" + tag + "_I.bin", Iall, (size_t)N * T * H);
  conn.assign((size_t)P * P, 0);
  for (int di = 0; di < ND; ++di) {
    Rqd[di].assign((size_t)H * P, 0.0);
    for (int i = 0; i < H; ++i) for (int j = 0; j < H; ++j) {
      double w = Wd[di][(size_t)i * H + j];
      Rqd[di][(size_t)i * P + j / C] += std::max(0.0, w);
      if (w != 0.0) conn[(i / C) * P + j / C] = 1;
    }
  }
  // single-thread reference with identical arithmetic
  std::vector<uint8_t> ref((size_t)N * (T + 1) * H, 0);
  for (int b = 0; b < N; ++b) {
    std::vector<double> v(H, 0.0), a(H, 0.0);
    uint8_t* spk = &ref[(size_t)b * (T + 1) * H];
    for (int t = 0; t < T - 1; ++t) for (int p = 0; p < P; ++p)
      core_step(p, &v[p * C], &a[p * C], spk, t, &Iall[((size_t)b * T + t + 1) * H], &spk[(size_t)(t + 1) * H + p * C]);
  }
  Rtot.assign(ND, std::vector<double>(H, 0.0));
  for (int di = 0; di < ND; ++di) for (int i = 0; i < H; ++i) {
    int p = i / C;
    for (int q = 0; q < P; ++q) if (q == p || conn[p * P + q]) Rtot[di][i] += Rqd[di][(size_t)i * P + q];
  }
  REFP = ref.data();
  double rtot = 0, rshort = 0;
  for (int di = 0; di < ND; ++di) {
    double rm = 0; for (int i = 0; i < H; ++i) for (int q = 0; q < P; ++q) rm += Rqd[di][(size_t)i * P + q];
    rm /= H; rtot += rm; if (DEL[di] < 4) rshort += rm;
  }
  printf("model=%s N=%d cores=%d delays=", tag.c_str(), N, P);
  for (int i = 0; i < ND; ++i) printf("%d%s", DEL[i], i + 1 < ND ? "," : "");
  printf(" d_min=%d R_total=%.4f R_short=%.4f budget=%.4f\n", DMIN, rtot, rshort, (1 - BETA) * THETA);
  int nconn = 0; for (int p = 0; p < P; ++p) for (int q = 0; q < P; ++q) nconn += (p != q) && conn[p * P + q];
  printf("inter-core edges (directed) = %d of %d\n", nconn, P * (P - 1));
  const int Ls[5] = {0, 5, 20, 100, 500};
  const int MAXMODE = argc > 3 ? atoi(argv[3]) : 2;
  for (int li = 0; li < 5; ++li) for (int mode = 0; mode < MAXMODE; ++mode) {
    Shared S; S.spk.assign(ref.size(), 0);
    S.pub.reset(new std::atomic<int64_t>[(size_t)N * P * (T + 1)]);
    for (size_t i = 0; i < (size_t)N * P * (T + 1); ++i) S.pub[i].store(i % (T + 1) == 0 ? 1 : 0);
    S.cert.assign((size_t)N * P * (T + 1), 0); S.tstart.assign(N * P, 0); S.tend.assign(N * P, 0);
    S.wait_ns.assign(P, 0); S.comp_ns.assign(P, 0);
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
    // per-core means over the whole run, in ms per sample, so they are comparable with median_ms
    double wsum = 0, csum = 0;
    for (int p = 0; p < P; ++p) { wsum += S.wait_ns[p]; csum += S.comp_ns[p]; }
    const double wms = wsum / P / N / 1e6, cms = csum / P / N / 1e6;
    printf("RESULT model=%s L_us=%d mode=%s median_ms=%.4f exact=%d cert_coverage=%.3f "
           "wait_ms=%.4f comp_ms=%.4f\n", tag.c_str(), Ls[li],
           mode == 0 ? "handshake" : (mode == 1 ? "cert" : "oracle"), ms[N / 2], (int)(S.spk == ref), tot ? (double)cov / tot : 0.0,
           wms, cms);
    fflush(stdout);
  }
}
