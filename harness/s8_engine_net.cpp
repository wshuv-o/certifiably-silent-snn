// NET-ENGINE-001: exact distributed engine over REAL TCP sockets.
//
// Every speed number in this project so far used *emulated* latency: published data was made visible to
// other cores only L microseconds after publication, inside one process. That is the single most
// damaging objection to the work -- the central claim is about synchronisation cost in a distributed
// system, and no distributed system was involved.
//
// This engine removes the emulation. Cores are partitioned across separate OS processes ("ranks") that
// exchange spikes and certificates over TCP. Latency, syscall cost, serialisation and scheduling are
// whatever the operating system and network actually impose. Run both ranks on one host for real
// loopback networking, or on two machines for real Ethernet, by pointing PEER at the other host.
//
// Protocol (fixed-size frames, so no parsing ambiguity):
//   struct Msg { int32 sample; int32 step; int32 cert[cores_per_rank]; uint8 spk[neurons_per_rank]; }
// The sample index is in the frame and spike buffers are allocated PER SAMPLE. An earlier version
// reset shared buffers at each sample boundary behind a barrier, which still raced: both ranks leave
// the barrier together, so one could send a sample-b+1 frame while the other was still zeroing, and
// that frame was erased. Two handshake runs reported exact=0 from precisely that. With per-sample
// slots there is no reset and no such race.
// After computing step t+1 each rank sends one frame. A reader thread consumes frames and publishes
// (step, cert) so the compute loop can decide whether it may proceed.
//
// Waiting rule, identical in substance to the shared-memory engine:
//   mode 0 (handshake): proceed when the peer has delivered step >= t+1-d_min  (the free lookahead the
//                       minimum synaptic delay provides -- the baseline is GIVEN this, as pre-registered)
//   mode 1 (cert):      also proceed when the peer's latest certificate covers t+1-d_min
//
// Correctness: rank 0 computes a single-process reference with identical arithmetic and compares the
// assembled spike trains bit-for-bit. exact=0 invalidates all timings.
//
// Build: g++ -O3 -march=native -std=c++20 -pthread s8_engine_net.cpp -o s8_engine_net
// Run:   rank 0:  s8_engine_net <dir> <tag> <mode> 0 2 <port>
//        rank 1:  s8_engine_net <dir> <tag> <mode> 1 2 <port> <host-of-rank-0>
#include <algorithm>
#include <atomic>
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

static int P, C, T, N, H, ND, DMIN;
static double BETA, THETA, RHO, GAMMA;
static std::vector<int> DEL;
static std::vector<std::vector<double>> Wd;
static std::vector<double> Iall;
static std::vector<std::vector<double>> Rqd;
static std::vector<uint8_t> conn;

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class X> static void readbin(const std::string& f, std::vector<X>& v, size_t n) {
  v.resize(n); std::ifstream in(f, std::ios::binary); in.read((char*)v.data(), n * sizeof(X));
  if ((size_t)in.gcount() != n * sizeof(X)) { fprintf(stderr, "short read %s\n", f.c_str()); exit(1); }
}

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

// ---- certificate (same exact/bounded split as the shared-memory engine) ----
// Cost matters as much as correctness: the naive bound (walk the weight row per neuron per horizon)
// made the certificate 2.3x slower than the handshake here, the same failure already fixed once in
// s6_engine_delays.cpp. Firing rate is ~4%, so build per-core lists of firing indices once per step
// and touch weights only at those positions.
struct FireLists {
  std::vector<std::vector<std::vector<int>>> fire;
  int base = 0, nr = 0;
};

static void build_fires(FireLists& F, const uint8_t* spk, int t) {
  int dmax = *std::max_element(DEL.begin(), DEL.end());
  int lo = std::max(0, t + 2 - dmax), hi = t + 1;
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

static inline double drive_bound(int p, int i, int t, int k, const FireLists& F, int peer_step,
                                 int lo_core, int hi_core) {
  double acc = 0.0;
  for (int di = 0; di < ND; ++di) {
    int ts = t + 1 + k - DEL[di];
    if (ts < 0) continue;
    const double* Wi = &Wd[di][(size_t)i * H];
    for (int q = 0; q < P; ++q) {
      if (q != p && !conn[p * P + q]) continue;
      bool mine = (q >= lo_core && q < hi_core);
      bool known = (ts <= t + 1) && (mine || ts <= peer_step);
      if (known) { for (int j : F.fire[ts - F.base][q]) acc += Wi[j]; }
      else       { acc += Rqd[di][(size_t)i * P + q]; }
    }
  }
  return acc;
}

static int cert_horizon(int p, int t, const double* v, const double* I, const FireLists& F,
                        int peer_step, int lo, int hi, int hmax) {
  int h = hmax;
  for (int ii = 0; ii < C && h > 0; ++ii) {
    int i = p * C + ii; double u = v[ii];
    for (int k = 1; k <= h; ++k) {
      int tau = t + 1 + k;
      if (tau > T - 1) { h = std::min(h, k - 1); break; }
      u = BETA * u + I[(size_t)tau * H + i] + drive_bound(p, i, t, k, F, peer_step, lo, hi);
      if (u >= THETA) { h = k - 1; break; }
    }
  }
  return h;
}

// ---- TCP plumbing ----
static int listen_accept(int port) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port);
  if (bind(srv, (sockaddr*)&a, sizeof(a)) < 0) { perror("bind"); exit(1); }
  listen(srv, 1);
  int fd = accept(srv, nullptr, nullptr);
  if (fd < 0) { perror("accept"); exit(1); }
  close(srv);
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));   // latency, not throughput
  return fd;
}
static int dial(const std::string& host, int port) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    if (connect(fd, (sockaddr*)&a, sizeof(a)) == 0) {
      int yes = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
      return fd;
    }
    close(fd); std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  fprintf(stderr, "could not connect to %s:%d\n", host.c_str(), port); exit(1);
}
static bool sendall(int fd, const void* p, size_t n) {
  const char* c = (const char*)p;
  while (n) { ssize_t k = send(fd, c, n, 0); if (k <= 0) return false; c += k; n -= (size_t)k; }
  return true;
}
static bool recvall(int fd, void* p, size_t n) {
  char* c = (char*)p;
  while (n) { ssize_t k = recv(fd, c, n, 0); if (k <= 0) return false; c += k; n -= (size_t)k; }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: %s <dir> <tag> <mode> <rank> <nranks> <port> [peer_host]\n", argv[0]); return 1; }
  std::string dir = argv[1], tag = argv[2];
  int mode = atoi(argv[3]), rank = atoi(argv[4]), nranks = atoi(argv[5]), port = atoi(argv[6]);
  std::string peer = argc > 7 ? argv[7] : "127.0.0.1";
  {
    std::ifstream d(dir + "/" + tag + "_dims.txt");
    d >> P >> C >> T >> N >> BETA >> THETA >> RHO >> GAMMA >> ND;
    DEL.resize(ND); for (int i = 0; i < ND; ++i) d >> DEL[i];
  }
  H = P * C; DMIN = *std::min_element(DEL.begin(), DEL.end());
  if (nranks != 2 || P % 2) { fprintf(stderr, "this build supports exactly 2 ranks and even P\n"); return 1; }
  Wd.resize(ND);
  for (int di = 0; di < ND; ++di) readbin(dir + "/" + tag + "_W" + std::to_string(di) + ".bin", Wd[di], (size_t)H * H);
  readbin(dir + "/" + tag + "_I.bin", Iall, (size_t)N * T * H);
  conn.assign((size_t)P * P, 0); Rqd.assign(ND, {});
  for (int di = 0; di < ND; ++di) {
    Rqd[di].assign((size_t)H * P, 0.0);
    for (int i = 0; i < H; ++i) for (int j = 0; j < H; ++j) {
      double w = Wd[di][(size_t)i * H + j];
      Rqd[di][(size_t)i * P + j / C] += std::max(0.0, w);
      if (w != 0.0) conn[(i / C) * P + j / C] = 1;
    }
  }
  const int cpr = P / 2, lo = rank * cpr, hi = lo + cpr, npr = cpr * C;
  const size_t MSG = 8 + 4 * (size_t)cpr + (size_t)npr;   // +4 for the sample index
  std::vector<char> txbuf(MSG), rxbuf(MSG);

  int fd = (rank == 0) ? listen_accept(port) : dial(peer, port);

  // shared state written by the reader thread
  std::vector<std::atomic<int>> peer_step(N);          // highest delivered step, per sample
  for (auto& x : peer_step) x.store(0);
  std::vector<std::atomic<int>> peer_cert((size_t)N * cpr);
  for (auto& c : peer_cert) c.store(0);
  std::vector<uint8_t> spk((size_t)N * (T + 1) * H, 0);  // per-sample slots: never reset, never raced
  std::atomic<bool> stop{false};

  std::thread reader([&] {
    while (!stop.load(std::memory_order_acquire)) {
      if (!recvall(fd, rxbuf.data(), MSG)) break;
      int sb, st; std::memcpy(&sb, rxbuf.data(), 4); std::memcpy(&st, rxbuf.data() + 4, 4);
      if (st == -1) break;                                 // end-of-stream sentinel
      int pl = (rank == 0) ? cpr : 0;                       // peer owns the other half
      std::memcpy(&spk[((size_t)sb * (T + 1) + st) * H + pl * C], rxbuf.data() + 8 + 4 * cpr, npr);
      for (int c = 0; c < cpr; ++c) { int cv; std::memcpy(&cv, rxbuf.data() + 8 + 4 * c, 4);
        peer_cert[(size_t)sb * cpr + c].store(cv, std::memory_order_release); }
      peer_step[sb].store(st, std::memory_order_release);
    }
  });

  std::vector<double> v(npr), a(npr);
  std::vector<int> myc(cpr, 0); FireLists F;
  std::vector<double> ms(N);
  for (int b = 0; b < N; ++b) {
    std::fill(v.begin(), v.end(), 0.0); std::fill(a.begin(), a.end(), 0.0);
    std::fill(myc.begin(), myc.end(), 0);
    const double* I = &Iall[(size_t)b * T * H];
    uint8_t* sb_spk = &spk[(size_t)b * (T + 1) * H];
    int64_t t0 = now_ns();
    for (int t = 0; t < T - 1; ++t) {
      const int need = t + 1 - DMIN;
      while (peer_step[b].load(std::memory_order_acquire) < need) {
        if (mode == 1) {
          bool ok = true;
          for (int c = 0; c < cpr; ++c)
            if (peer_cert[(size_t)b * cpr + c].load(std::memory_order_acquire) < need) { ok = false; break; }
          if (ok) break;
        }
        std::this_thread::yield();
      }
      for (int p = lo; p < hi; ++p)
        core_step(p, &v[(p - lo) * C], &a[(p - lo) * C], sb_spk, t, &I[(size_t)(t + 1) * H],
                  &sb_spk[(size_t)(t + 1) * H + p * C]);
      if (mode == 1) {
        int ps = peer_step[b].load(std::memory_order_acquire);
        bool any = false;
        for (int p = lo; p < hi && !any; ++p) {
          for (int ii = 0; ii < C; ++ii) if (sb_spk[(size_t)(t + 1) * H + p * C + ii]) { any = true; break; }
          if (t + 1 >= myc[p - lo]) any = true;
        }
        if (any) build_fires(F, sb_spk, t);
        for (int p = lo; p < hi; ++p) {
          bool spiked = false;
          for (int ii = 0; ii < C; ++ii) if (sb_spk[(size_t)(t + 1) * H + p * C + ii]) { spiked = true; break; }
          if (spiked || t + 1 >= myc[p - lo])
            myc[p - lo] = t + 1 + cert_horizon(p, t, &v[(p - lo) * C], I, F, ps, lo, hi, 16);
        }
      }
      int st = t + 1; std::memcpy(txbuf.data(), &b, 4); std::memcpy(txbuf.data() + 4, &st, 4);
      for (int c = 0; c < cpr; ++c) std::memcpy(txbuf.data() + 8 + 4 * c, &myc[c], 4);
      std::memcpy(txbuf.data() + 8 + 4 * cpr, &sb_spk[(size_t)(t + 1) * H + lo * C], npr);
      if (!sendall(fd, txbuf.data(), MSG)) { fprintf(stderr, "send failed\n"); return 1; }
    }
    // No barrier and no reset are needed: each sample has its own buffer slot, so a frame that
    // arrives early simply lands in the slot it belongs to. Still wait for the peer to finish this
    // sample so the per-sample timing is comparable between ranks.
    while (peer_step[b].load(std::memory_order_acquire) < T - 1) std::this_thread::yield();
    ms[b] = (now_ns() - t0) / 1e6;
  }
  int zero = 0, sentinel = -1;
  std::memcpy(txbuf.data(), &zero, 4); std::memcpy(txbuf.data() + 4, &sentinel, 4);
  sendall(fd, txbuf.data(), MSG);
  stop.store(true, std::memory_order_release);
  shutdown(fd, SHUT_RDWR); reader.join(); close(fd);

  std::sort(ms.begin(), ms.end());
  if (rank == 0) {
    // single-process reference with identical arithmetic, on the last sample's assembled state
    std::vector<uint8_t> ref((size_t)(T + 1) * H, 0);
    std::vector<double> rv(H, 0.0), ra(H, 0.0);
    const double* I = &Iall[(size_t)(N - 1) * T * H];
    for (int t = 0; t < T - 1; ++t) for (int p = 0; p < P; ++p)
      core_step(p, &rv[p * C], &ra[p * C], ref.data(), t, &I[(size_t)(t + 1) * H], &ref[(size_t)(t + 1) * H + p * C]);
    bool exact = (std::memcmp(ref.data(), &spk[(size_t)(N - 1) * (T + 1) * H], ref.size()) == 0);
    printf("NETRESULT tag=%s mode=%s ranks=%d cores=%d median_ms=%.4f exact=%d\n",
           tag.c_str(), mode == 0 ? "handshake" : "cert", nranks, P, ms[N / 2], (int)exact);
  }
  fflush(stdout);
  return 0;
}
