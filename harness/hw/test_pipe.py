"""PipeCU (pipelined) vs the reference recursion on random operands: every output equal after latency K+1."""
import sys, os, random
sys.argv = [sys.argv[0], "t", sys.argv[1] if len(sys.argv) > 1 else "4", "20"]
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
src = open(os.path.join(os.path.dirname(__file__), "pnr.py")).read().split("\n\nname, K, clk =")[0]
ns = {"__file__": __file__}; exec(src, ns)
from amaranth.sim import Simulator
K = int(sys.argv[2]); W = 24; F = 12; BETA, THETA, RHO, GAMMA = 2484, 4096, 3819, 82
dut = ns["PipeCU"](K)

def ref(v, a, s, bm, x, b):
    u, ak, fire = v, a, 0; out = {}
    for k in range(K):
        u = ((u * BETA) >> F) + x[k] + (b[k] if (bm and k > 0) else 0)
        ak = ((ak * RHO) >> F) + (GAMMA if (s and k == 0) else 0)
        f = int(u >= THETA + ak); fire |= f
        if k == 0: out.update(s=f, v=u - (THETA + ak if f else 0), a=ak)
        if k == max(K - 2, 0): out.update(vk=u, ak=ak)
    out["fire"] = fire; return out

random.seed(1); N = 3000
vecs = [(random.randint(-300000, 300000), random.randint(0, 1100), random.randint(0, 1), random.randint(0, 1),
         [random.randint(-60000, 60000) for _ in range(K)], [random.randint(0, 3000) for _ in range(K)]) for _ in range(N)]
errs = []
async def tb(ctx):
    ctx.set(dut.beta, BETA); ctx.set(dut.rho, RHO); ctx.set(dut.theta, THETA); ctx.set(dut.gamma, GAMMA)
    for t in range(N + K + 1):
        if t < N:
            v, a, s, bm, x, b = vecs[t]
            ctx.set(dut.i_v, v); ctx.set(dut.i_a, a); ctx.set(dut.i_s, s); ctx.set(dut.i_bmask, bm)
            for k in range(K): ctx.set(dut.i_x[k], x[k]); ctx.set(dut.i_b[k], b[k])
        await ctx.tick()
        j = t - K                                              # outputs valid K+1 edges after the inputs
        if 0 <= j < N:
            r = ref(*vecs[j])
            hw = dict(s=ctx.get(dut.o_s), v=ctx.get(dut.o_v), a=ctx.get(dut.o_a), fire=ctx.get(dut.o_fire), vk=ctx.get(dut.o_vk), ak=ctx.get(dut.o_ak))
            if hw != r: errs.append((j, hw, r))
sim = Simulator(dut); sim.add_clock(1e-8); sim.add_testbench(tb); sim.run()
print(f"RESULT PipeCU K={K} vectors={N} mismatches={len(errs)}", errs[:2])
