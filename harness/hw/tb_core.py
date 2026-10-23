"""Cycle-level co-simulation of one Core against the fixed-point golden model.

The core is driven with the network's real spike stream (golden spikes of all H neurons, routed to the core only
when the presynaptic neuron has a synapse in it) and its host input. Checks at every step: the core's spikes
equal the golden spikes of its neurons; at every normal pass with the certificate enabled, the certificate bit
equals the golden certificate of the core; after every executed pass, the membrane and adaptation state equal
the golden state. Reports steps by pass kind and core cycles.
Usage: python tb_core.py <data_dir> <model> <core> <samples> [C]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amaranth.sim import Simulator
from golden import Net, q
from energy_estimate import load
from core import Core
from core_lp import CoreLP
import lowcert as LC
LP = bool(os.environ.get("LP"))

d, model, p, NS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
C = int(sys.argv[5]) if len(sys.argv) > 5 else 32
K, W = 4, 24
net = load(d, model)
g = Net(net["W"], net["D"], net["I"], net["beta"], net["theta"], net["rho"], net["gamma"], C)
H, T, D = g.H, net["T"], g.D
rows = slice(p * C, (p + 1) * C)
weights = [[int(x) for x in w[rows, :].T.reshape(-1)] for w in g.W]       # index pre*C + post
bounded = [[int(x) for x in np.broadcast_to(g.bounded(k), (H,))[rows]] for k in range(1, K + 1)]
routed = np.array([any(np.any(w[rows, j] != 0) for w in g.W) for j in range(H)])
consts = (int(g.beta), int(g.theta), int(g.rho), int(g.gamma))

tot = dict(normal=0, catch=0, idle=0, cert_checked=0, cert_true=0, cycles=0, pass_cycles=0, ev_cycles=0, in_cycles=0)
for b in range(NS):
    S, cert, viol, V, A, slots_g = g.run(b, K=K, adapt=True)
    assert viol == 0
    if LP:                                    # expected certificate: reduced-precision bound of the core's neurons
        cert_x = np.zeros_like(cert)
        for t in range(1, T - K):
            for pp in range(g.P):
                cert_x[t, pp] = not any(LC.ref(int(V[t, i]), int(A[t, i]), int(S[t, i]),
                                               [int(slots_g[t, k, i] + g.I[b, t + 1 + k, i]) for k in range(K)],
                                               [int(np.broadcast_to(g.bounded(k + 1), (H,))[i]) for k in range(K)])
                                        for i in range(pp * C, (pp + 1) * C))
        assert not (cert_x & ~cert).any(), "reduced-precision certificate not contained in exact certificate"
        cert = cert_x
    dut = (CoreLP if LP else Core)(C, H, K, D, W, *consts, weights, bounded)
    errs = []

    async def tb(ctx):
        cyc = 0

        async def tick(kind):
            nonlocal cyc
            await ctx.tick(); cyc += 1; tot[kind] += 1

        async def host(step):                                    # add host input of `step` into its slot
            for i in range(C):
                ctx.set(dut.in_valid, 1); ctx.set(dut.in_slot, step % 16); ctx.set(dut.in_i, i)
                ctx.set(dut.in_val, int(g.I[b, step, p * C + i]))
                await tick("in_cycles")
            ctx.set(dut.in_valid, 0)

        for s in range(1, min(K, T)):
            await host(s)
        for tau in range(1, T):
            if tau + K - 1 < T:
                await host(tau + K - 1)
            ctx.set(dut.cert_en, int(tau >= 2 and tau - 1 + K < T)); ctx.set(dut.go, 1)
            await tick("pass_cycles"); ctx.set(dut.go, 0)
            while not ctx.get(dut.done):
                await tick("pass_cycles")
            kind = ctx.get(dut.kind); spk = ctx.get(dut.spk)
            gs = sum(int(S[tau, p * C + i]) << i for i in range(C))
            if spk != gs:
                errs.append(f"spikes tau={tau} kind={kind}")
            tot[["normal", "catch", "idle"][kind]] += 1
            if kind == 0 and tau >= 2 and tau - 1 + K < T:
                tot["cert_checked"] += 1; tot["cert_true"] += int(ctx.get(dut.cert_ok))
                if bool(ctx.get(dut.cert_ok)) != bool(cert[tau - 1, p]):
                    errs.append(f"cert tau={tau} hw={ctx.get(dut.cert_ok)} golden={cert[tau - 1, p]}")
            await tick("pass_cycles")                              # FIN -> IDLE
            if kind != 2:
                for i in range(C):
                    hv, ha = ctx.get(dut.mem_v.data[i]), ctx.get(dut.mem_a.data[i])
                    if hv != V[tau, p * C + i] or ha != A[tau, p * C + i]:
                        errs.append(f"state tau={tau} kind={kind} i={i} v={hv}/{V[tau, p * C + i]} a={ha}/{A[tau, p * C + i]}")
                        break
            for j in np.nonzero(S[tau] & routed)[0]:
                ctx.set(dut.ev_valid, 1); ctx.set(dut.ev_pre, int(j))
                await tick("ev_cycles"); ctx.set(dut.ev_valid, 0)
                while not ctx.get(dut.ready):
                    await tick("ev_cycles")
        tot["cycles"] += cyc

    sim = Simulator(dut); sim.add_clock(1e-8); sim.add_testbench(tb); sim.run()
    print(f"sample {b}: errors={len(errs)} {errs[:5]}", flush=True)
    if errs:
        sys.exit(1)

steps = tot["normal"] + tot["catch"] + tot["idle"]
print(f"RESULT model={model} core={p} C={C} samples={NS} steps={steps} normal={tot['normal']} catchup={tot['catch']} "
      f"idle={tot['idle']} cert_checked={tot['cert_checked']} cert_true={tot['cert_true']} "
      f"neuron_passes_saved={1 - (tot['normal'] + tot['catch']) / steps:.3f} cycles={tot['cycles']} "
      f"pass_cycles={tot['pass_cycles']} event_cycles={tot['ev_cycles']} input_cycles={tot['in_cycles']}")
