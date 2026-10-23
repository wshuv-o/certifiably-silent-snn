"""Core with the reduced-precision certificate (HW-008). Same memories, synapse unit and host interface as
core.Core; differences:
  - the neuron unit is CertUnit(K=1) (exact Q12 update);
  - the certificate is evaluated on the reduced-precision bound of lowcert.py (16-bit, directed rounding,
    saturating) from the same state and window reads;
  - catch-up advances each neuron through the K-1 skipped steps in K-1 consecutive cycles of the neuron unit,
    holding the neuron's state in registers: one state read and one write per neuron.
"""
from amaranth import *
from amaranth.lib.memory import Memory
from core import CertUnit, F
import lowcert as LC


class CoreLP(Elaboratable):
    def __init__(self, C, H, K, D, W, beta, theta, rho, gamma, weights, bounded, R=16):
        assert R % K == 0 and R > max(D) + K and K == LC.K
        self.C, self.H, self.K, self.D, self.W, self.R = C, H, K, list(D), W, R
        self.consts = (beta, theta, rho, gamma); self.weights, self.bounded = weights, bounded
        self.go = Signal(); self.cert_en = Signal(); self.done = Signal(); self.ready = Signal()
        self.spk = Signal(C); self.cert_ok = Signal(); self.kind = Signal(2)
        self.ev_valid = Signal(); self.ev_pre = Signal(range(H))
        self.in_valid = Signal(); self.in_slot = Signal(range(R)); self.in_i = Signal(range(C))
        self.in_val = Signal(signed(W))
        self.tau = Signal(16)

    def lowprec_fire(self, m, v, a, s, xs, bs):
        """Combinational reduced-precision bound over K steps; returns 'a neuron can fire'."""
        U = LC.sat(m, LC.cshr(v, LC.S), "lu0"); A = a >> LC.S; fire = Const(0, 1)
        for k in range(self.K):
            c = Mux(U < 0, LC.BD, LC.BU)
            nu = LC.sat(m, LC.cshr(U * c, LC.PB) + LC.cshr(xs[k], LC.S) + LC.cshr(bs[k], LC.S), f"lu{k + 1}")
            na = Signal(signed(LC.WL), name=f"la{k + 1}")
            m.d.comb += na.eq(((A * LC.RD) >> LC.PB) + (Mux(s, LC.GM, 0) if k == 0 else 0))
            fire = fire | (nu >= LC.TH + na)
            U, A = nu, na
        return fire

    def elaborate(self, platform):
        m = Module(); C, K, R, W, D = self.C, self.K, self.R, self.W, self.D
        rows = R // K
        m.submodules.nu = nu = CertUnit(1, W, *self.consts)
        mv = Memory(shape=signed(W), depth=C, init=[]); ma = Memory(shape=signed(W), depth=C, init=[])
        ms = Memory(shape=1, depth=C, init=[])
        m.submodules += [mv, ma, ms]; self.mem_v, self.mem_a = mv, ma
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
        j = Signal(range(K)); hv = Signal(signed(W)); ha = Signal(signed(W)); hs = Signal()   # catch-up holding regs

        for b in range(K):
            off = Signal(range(K), name=f"off{b}")
            m.d.comb += off.eq((b - base) % K)
            m.d.comb += brd[b].addr.eq((((base + off) % R) // K) * C + i)
        xs = [Signal(signed(W), name=f"xw{k}") for k in range(K)]
        bdata = Array([p.data for p in brd])
        for k in range(K):
            m.d.comb += [xs[k].eq(bdata[(base + k) % K]), brom[k].addr.eq(i)]
        for p in (rv, ra, rs, wv, wa, ws):
            m.d.comb += p.addr.eq(i)
        # neuron unit operands: memory (normal pass, catch-up sub-step 0) or holding registers (sub-steps >= 1)
        held = (mode == 1) & (j != 0)
        m.d.comb += [nu.v.eq(Mux(held, hv, rv.data)), nu.a.eq(Mux(held, ha, ra.data)), nu.s.eq(Mux(held, hs, rs.data)),
                     nu.x[0].eq(bdata[(base + j) % K]), nu.b[0].eq(0), nu.bmask.eq(0)]
        lp_fire = self.lowprec_fire(m, rv.data, ra.data, rs.data, xs, [p.data for p in brom])
        m.d.comb += [self.ready.eq(0), self.done.eq(0),
                     self.cert_ok.eq((mode == 0) & en & ~fire_any), self.kind.eq(mode)]

        def rmw(slot, idx, val):
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
                    m.d.sync += [self.tau.eq(self.tau + 1), en.eq(self.cert_en), fire_any.eq(0), self.spk.eq(0),
                                 i.eq(0), j.eq(0)]
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
                    m.d.comb += [wv.data.eq(nu.v_new), wa.data.eq(nu.ak[0]), ws.data.eq(nu.s_new),
                                 wv.en.eq(1), wa.en.eq(1), ws.en.eq(1)]
                    m.d.sync += [fire_any.eq(fire_any | lp_fire), self.spk.eq(self.spk | (nu.s_new << i))]
                    for b in range(K):                                  # clear slot base
                        m.d.comb += [bwr[b].addr.eq(brd[b].addr), bwr[b].data.eq(0), bwr[b].en.eq(((b - base) % K) == 0)]
                    m.d.sync += i.eq(i + 1)
                    with m.If(i == C - 1):
                        m.next = "FIN"
                with m.Else():                                         # catch-up: sub-step j of neuron i
                    m.d.sync += [hv.eq(nu.u[0]), ha.eq(nu.ak[0]), hs.eq(0)]
                    for b in range(K):                                  # clear the slot just consumed
                        m.d.comb += [bwr[b].addr.eq(brd[b].addr), bwr[b].data.eq(0),
                                     bwr[b].en.eq(((b - base) % K) == j)]
                    with m.If(j == K - 2):
                        m.d.comb += [wv.data.eq(nu.u[0]), wa.data.eq(nu.ak[0]), ws.data.eq(0),
                                     wv.en.eq(1), wa.en.eq(1), ws.en.eq(1)]
                        m.d.sync += [j.eq(0), i.eq(i + 1)]
                        with m.If(i == C - 1):
                            m.next = "FIN"
                    with m.Else():
                        m.d.sync += j.eq(j + 1)
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
