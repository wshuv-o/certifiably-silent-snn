"""Amaranth model of one time-multiplexed neuromorphic core of C fixed-point ALIF neurons with delay taps D and a
K-step silence certificate unit.

Datapath. The certificate unit (CertUnit) evaluates, for one neuron per clock, the recursion
    u_k = floor(beta * u_{k-1}) + x_k + b_k,  a_k = floor(rho * a_{k-1}) + [k = 1] gamma s,  f_k = u_k >= theta + a_k
for k = 1..K, where x_k is the delay accumulator slot of step base+k-1 (host input already added) and b_k the
bounded drive sum_{d<k} R^(d). Stage 1 with b_1 = 0 is the neuron update itself (spike s' = f_1, reset
v' = u_1 - s' (theta + a_1)), so the neuron unit is the first stage of the certificate unit.

Schedule. Per algorithmic step tau the host raises `go` and the core runs one of three passes over its neurons:
  normal   neuron update of step tau; in parallel, the certificate rooted at tau-1 (window tau..tau+K-1), using the
           same state and accumulator reads. If no neuron of the core can fire in the window, steps
           tau+1..tau+K-2 are skipped and step tau+K-1 runs a catch-up pass.
  catch-up stages 1..K-1 with b = 0 advance every neuron from tau to tau+K-1 in one pass (exact, as the core is
           silent), and clear the K-1 accumulator slots consumed.
  idle     no neuron work.
After the pass the host delivers the step's spike events (`ev_pre`); the synapse unit adds the weights of each
tap into the accumulator slot tau+d. Accumulator slots live in a ring of R steps split into K banks by slot mod K,
so the K consecutive slots of a window are read in one cycle.
"""
from amaranth import *
from amaranth.lib.memory import Memory

F = 12


class CertUnit(Elaboratable):
    """Combinational K-stage certificate datapath (one neuron)."""

    def __init__(self, K, W, beta, theta, rho, gamma):
        self.K, self.W = K, W
        self.beta, self.theta, self.rho, self.gamma = beta, theta, rho, gamma
        self.v = Signal(signed(W)); self.a = Signal(signed(W)); self.s = Signal()
        self.x = [Signal(signed(W), name=f"x{k}") for k in range(1, K + 1)]
        self.b = [Signal(signed(W), name=f"b{k}") for k in range(1, K + 1)]
        self.bmask = Signal()                                   # 1: include bounded drive (certificate)
        self.u = [Signal(signed(W), name=f"u{k}") for k in range(1, K + 1)]
        self.ak = [Signal(signed(W), name=f"a{k}") for k in range(1, K + 1)]
        self.fire = Signal(K)
        self.s_new = Signal(); self.v_new = Signal(signed(W))

    def elaborate(self, platform):
        m = Module(); F_ = F
        u_prev, a_prev = self.v, self.a
        for k in range(self.K):
            b = Mux(self.bmask, self.b[k], 0) if k > 0 else 0
            m.d.comb += self.u[k].eq(((u_prev * self.beta) >> F_) + self.x[k] + b)
            g = Mux(self.s, self.gamma, 0) if k == 0 else 0
            m.d.comb += self.ak[k].eq(((a_prev * self.rho) >> F_) + g)
            m.d.comb += self.fire[k].eq(self.u[k] >= self.theta + self.ak[k])
            u_prev, a_prev = self.u[k], self.ak[k]
        m.d.comb += [self.s_new.eq(self.fire[0]),
                     self.v_new.eq(self.u[0] - Mux(self.fire[0], self.theta + self.ak[0], 0))]
        return m


class Core(Elaboratable):
    def __init__(self, C, H, K, D, W, beta, theta, rho, gamma, weights, bounded, R=16):
        """weights[t]: list of H*C ints, index pre*C + post (tap t); bounded[k]: list of C ints (b_{k+1})."""
        assert R % K == 0 and R > max(D) + K
        self.C, self.H, self.K, self.D, self.W, self.R = C, H, K, list(D), W, R
        self.consts = (beta, theta, rho, gamma); self.weights, self.bounded = weights, bounded
        self.go = Signal(); self.cert_en = Signal(); self.done = Signal(); self.ready = Signal()
        self.spk = Signal(C); self.cert_ok = Signal(); self.kind = Signal(2)    # 0 normal, 1 catch-up, 2 idle
        self.ev_valid = Signal(); self.ev_pre = Signal(range(H))
        self.in_valid = Signal(); self.in_slot = Signal(range(R)); self.in_i = Signal(range(C))
        self.in_val = Signal(signed(W))
        self.tau = Signal(16)

    def elaborate(self, platform):
        m = Module(); C, K, R, W, D = self.C, self.K, self.R, self.W, self.D
        rows = R // K
        m.submodules.cu = cu = CertUnit(K, W, *self.consts)
        mv = Memory(shape=signed(W), depth=C, init=[]); ma = Memory(shape=signed(W), depth=C, init=[])
        ms = Memory(shape=1, depth=C, init=[])
        m.submodules += [mv, ma, ms]
        self.mem_v, self.mem_a = mv, ma
        rv, ra, rs = (x.read_port(domain="comb") for x in (mv, ma, ms))
        wv, wa, ws = (x.write_port() for x in (mv, ma, ms))
        banks = [Memory(shape=signed(W), depth=rows * C, init=[]) for _ in range(K)]
        m.submodules += banks
        brd = [b.read_port(domain="comb") for b in banks]; bwr = [b.write_port() for b in banks]
        wmem = [Memory(shape=signed(W), depth=self.H * C, init=w) for w in self.weights]
        m.submodules += wmem
        wrd = [x.read_port(domain="comb") for x in wmem]
        bmem = [Memory(shape=signed(W), depth=C, init=b) for b in self.bounded]
        m.submodules += bmem
        brom = [x.read_port(domain="comb") for x in bmem]

        i = Signal(range(C)); base = Signal(16); mode = Signal(2); en = Signal(); fire_any = Signal()
        skip = Signal(range(K + 1)); pre = Signal(range(self.H)); tap = Signal(range(len(D))); post = Signal(range(C))

        def slot_addr(slot, idx):                              # (bank, address) of accumulator word
            s = slot % R
            return s % K, (s // K) * C + idx

        # window reads: bank b holds the window slot base + ((b - base) mod K)
        for b in range(K):
            off = Signal(range(K), name=f"off{b}")
            m.d.comb += off.eq((b - base) % K)
            m.d.comb += brd[b].addr.eq((((base + off) % R) // K) * C + i)
        for k in range(K):
            m.d.comb += cu.x[k].eq(Array([p.data for p in brd])[(base + k) % K])
            m.d.comb += [brom[k].addr.eq(i), cu.b[k].eq(brom[k].data)]
        for p in (rv, ra, rs, wv, wa, ws):
            m.d.comb += p.addr.eq(i)
        m.d.comb += [cu.v.eq(rv.data), cu.a.eq(ra.data), cu.s.eq(rs.data), cu.bmask.eq(mode == 0)]
        m.d.comb += [self.ready.eq(0), self.done.eq(0)]
        m.d.comb += [self.cert_ok.eq((mode == 0) & en & ~fire_any), self.kind.eq(mode)]

        def rmw(slot, idx, val):                               # accumulator read-modify-write, one word
            s = Signal(range(R), name="rmw_slot"); bank = Signal(range(K), name="rmw_bank")
            addr = Signal(range(rows * C), name="rmw_addr")
            m.d.comb += [s.eq(slot % R), bank.eq(s % K), addr.eq((s // K) * C + idx)]
            for b in range(K):
                with m.If(bank == b):
                    m.d.comb += [brd[b].addr.eq(addr), bwr[b].addr.eq(addr),
                                 bwr[b].data.eq(brd[b].data + val), bwr[b].en.eq(1)]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                with m.If(self.in_valid):
                    rmw(self.in_slot, self.in_i, self.in_val)
                with m.Elif(self.ev_valid):
                    m.d.sync += [pre.eq(self.ev_pre), tap.eq(0), post.eq(0)]
                    m.next = "EV"
                with m.Elif(self.go):
                    m.d.sync += [self.tau.eq(self.tau + 1), en.eq(self.cert_en), fire_any.eq(0), self.spk.eq(0), i.eq(0)]
                    with m.If(skip == 0):
                        m.d.sync += [mode.eq(0), base.eq(self.tau + 1)]
                        m.next = "PASS"
                    with m.Elif(skip == 1):
                        m.d.sync += [mode.eq(1), base.eq(self.tau + 1 - (K - 2)), skip.eq(0)]
                        m.next = "PASS"
                    with m.Else():
                        m.d.sync += [mode.eq(2), skip.eq(skip - 1)]
                        m.next = "FIN"
            with m.State("PASS"):
                with m.If(mode == 0):
                    m.d.comb += [wv.data.eq(cu.v_new), wa.data.eq(cu.ak[0]), ws.data.eq(cu.s_new)]
                    m.d.sync += [fire_any.eq(fire_any | cu.fire.any()), self.spk.eq(self.spk | (cu.s_new << i))]
                with m.Else():
                    m.d.comb += [wv.data.eq(cu.u[K - 2]), wa.data.eq(cu.ak[K - 2]), ws.data.eq(0)]
                m.d.comb += [wv.en.eq(1), wa.en.eq(1), ws.en.eq(1)]
                for b in range(K):                             # clear consumed slots (1 normal, K-1 catch-up)
                    off = (b - base) % K
                    m.d.comb += [bwr[b].addr.eq(brd[b].addr), bwr[b].data.eq(0),
                                 bwr[b].en.eq(Mux(mode == 0, off == 0, off < K - 1))]
                m.d.sync += i.eq(i + 1)
                with m.If(i == C - 1):
                    m.next = "FIN"
            with m.State("FIN"):
                m.d.comb += self.done.eq(1)
                with m.If((mode == 0) & en & ~fire_any):
                    m.d.sync += skip.eq(K - 1)
                m.next = "IDLE"
            with m.State("EV"):
                dsel = Array([Const(d, 16) for d in D])[tap]
                m.d.comb += [p.addr.eq(pre * C + post) for p in wrd]
                rmw(self.tau + dsel, post, Array([p.data for p in wrd])[tap])
                m.d.sync += post.eq(post + 1)
                with m.If(post == C - 1):
                    m.d.sync += [post.eq(0), tap.eq(tap + 1)]
                    with m.If(tap == len(D) - 1):
                        m.next = "IDLE"
        return m
