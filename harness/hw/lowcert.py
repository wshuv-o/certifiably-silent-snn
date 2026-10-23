"""HW-007: reduced-precision certificate unit (directed rounding, saturating), K pipelined stages.
Stage k receives its accumulator operand x_k and bounded drive b_k in the cycle it computes (bank reads skewed by
k cycles), so no operand is carried down the pipeline. `en` low freezes all stage registers (early abort).
Operands: U in WL-bit signed units of 2^S (Q12), decay factors with PB fractional bits.
Usage: python lowcert.py check <data_dir> <model> <ncycles>   (RTL vs reference on real operands)
       python lowcert.py rtlil <out.il>"""
import os, sys, random
import numpy as np
from amaranth import *
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
F = 12; K = 4; S = 4; PB = 6; WL = 16; W = 24
BETA, THETA, RHO, GAMMA = 2484, 4096, 3819, 82
BU = -((-BETA) >> (F - PB)); BD = BETA >> (F - PB); RD = RHO >> (F - PB); TH = THETA >> S; GM = GAMMA >> S
LO, HI = -(1 << (WL - 1)), (1 << (WL - 1)) - 1


def sat(m, x, name):
    y = Signal(signed(WL), name=name)
    with m.If(x > HI):
        m.d.comb += y.eq(HI)
    with m.Elif(x < LO):
        m.d.comb += y.eq(LO)
    with m.Else():
        m.d.comb += y.eq(x)
    return y


def cshr(x, s):                                     # ceil(x / 2^s) for signed x
    return -((-x) >> s)


class LowCU(Elaboratable):
    def __init__(self):
        self.bu = Signal(PB + 1); self.bd = Signal(PB + 1); self.rd = Signal(PB + 1)
        self.th = Signal(signed(WL)); self.gm = Signal(signed(WL))
        self.en = Signal()
        self.i_v = Signal(signed(W)); self.i_a = Signal(signed(W)); self.i_s = Signal()
        self.i_x = [Signal(signed(W), name=f"i_x{k}") for k in range(K)]   # x_k valid k cycles after i_v
        self.i_b = [Signal(signed(W), name=f"i_b{k}") for k in range(K)]
        self.o_fire = Signal()

    def ports(self):
        return [self.bu, self.bd, self.rd, self.th, self.gm, self.en, self.i_v, self.i_a, self.i_s, *self.i_x,
                *self.i_b, self.o_fire]

    def elaborate(self, platform):
        m = Module()
        U = Signal(signed(WL)); A = Signal(signed(WL)); sb = Signal(); fire = Const(0, 1)
        with m.If(self.en):
            m.d.sync += [U.eq(sat(m, cshr(self.i_v, S), "u0s")), A.eq(self.i_a >> S), sb.eq(self.i_s)]
        for k in range(K):
            c = Mux(U < 0, self.bd, self.bu)
            prod = Signal(signed(WL + PB + 2), name=f"prod{k}")
            m.d.comb += prod.eq(U * c)
            nu = sat(m, cshr(prod, PB) + cshr(self.i_x[k], S) + cshr(self.i_b[k], S), f"usat{k}")
            na = Signal(signed(WL), name=f"na{k}")
            m.d.comb += na.eq(((A * self.rd) >> PB) + (Mux(sb, self.gm, 0) if k == 0 else 0))
            Ur, Ar, fr = Signal(signed(WL), name=f"U{k+1}"), Signal(signed(WL), name=f"A{k+1}"), Signal(name=f"f{k+1}")
            with m.If(self.en):
                m.d.sync += [Ur.eq(nu), Ar.eq(na), fr.eq(fire | (nu >= self.th + na))]
            if k == 0:
                sb2 = sb                                    # gamma term only in stage 1
            U, A, fire = Ur, Ar, fr
        m.d.comb += self.o_fire.eq(fire)
        return m


def ref(v, a, s, x, b):
    """Reference of the unit's arithmetic (saturating, directed rounding)."""
    sat_ = lambda y: max(LO, min(HI, y))
    U = sat_(cshr(v, S)); A = a >> S; fire = 0
    for k in range(K):
        c = BD if U < 0 else BU
        U = sat_(cshr(U * c, PB) + cshr(x[k], S) + cshr(b[k], S))
        A = ((A * RD) >> PB) + (GM if (s and k == 0) else 0)
        fire |= int(U >= TH + A)
    return fire


def operands(ddir, model, n):
    from golden import Net
    from energy_estimate import load
    net = load(ddir, model)
    g = Net(net["W"], net["D"], net["I"], net["beta"], net["theta"], net["rho"], net["gamma"], 32)
    Bk = [np.broadcast_to(g.bounded(k + 1), (g.H,)) for k in range(K)]
    ops = []
    for bb in range(net["N"]):
        Sp, cert, viol, V, A, slots = g.run(bb, K=K)
        for t in range(1, Sp.shape[0] - K):
            for i in range(g.H):
                ops.append((int(V[t, i]), int(A[t, i]), int(Sp[t, i]),
                            [int(slots[t, k, i] + g.I[bb, t + 1 + k, i]) for k in range(K)], [int(Bk[k][i]) for k in range(K)]))
                if len(ops) >= n: return ops
    return ops


if __name__ == "__main__":
    if sys.argv[1] == "rtlil":
        from amaranth.back import rtlil
        u = LowCU(); open(sys.argv[2], "w").write(rtlil.convert(u, ports=u.ports(), name="top")); sys.exit(0)
    from amaranth.sim import Simulator
    ops = operands(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    random.seed(2)
    ops += [(random.randint(-(1 << 23), (1 << 23) - 1), random.randint(0, 1300), random.randint(0, 1),   # a <= gamma/(1-rho) ~ 1210
             [random.randint(-(1 << 23), (1 << 23) - 1) for _ in range(K)], [random.randint(0, 1 << 22) for _ in range(K)])
            for _ in range(2000)]                         # saturation corners
    dut = LowCU(); errs = []; N = len(ops)

    async def tb(ctx):
        for sig, val in ((dut.bu, BU), (dut.bd, BD), (dut.rd, RD), (dut.th, TH), (dut.gm, GM), (dut.en, 1)):
            ctx.set(sig, val)
        for t in range(N + K + 1):
            if t < N:
                v, a, s, x, b = ops[t]; ctx.set(dut.i_v, v); ctx.set(dut.i_a, a); ctx.set(dut.i_s, s)
            for k in range(K):                            # stage k works on the vector issued k cycles earlier
                j = t - k - 1
                if 0 <= j < N:
                    ctx.set(dut.i_x[k], ops[j][3][k]); ctx.set(dut.i_b[k], ops[j][4][k])
            await ctx.tick()
            j = t - K
            if 0 <= j < N and ctx.get(dut.o_fire) != ref(*ops[j]):
                errs.append(j)
    sim = Simulator(dut); sim.add_clock(1e-8); sim.add_testbench(tb); sim.run()
    real = sum(1 for o in ops[:N - 2000] if True)
    print("first errors", errs[:5], "real-data errors", sum(1 for j in errs if j < N - 2000))
    print(f"RESULT LowCU K={K} S={S} PB={PB} WL={WL} vectors={N} (real {N - 2000}, random 2000) mismatches={len(errs)}")
