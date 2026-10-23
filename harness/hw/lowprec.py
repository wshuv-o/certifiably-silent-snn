"""HW-007: reduced-precision certificate with directed rounding.
The certificate only needs an upper bound on the membrane and a lower bound on the threshold, so it can run on
coarse operands: membrane, input and bounded drive are rounded up to multiples of 2^s (Q12 units), the decay factor
is rounded to pb fractional bits (up for a non-negative operand, down for a negative one), the adaptation is rounded
down. Every step is monotone, hence U_k * 2^s >= u_k and Thr_k * 2^s <= theta + a_k, and a core certified at reduced
precision is certified exactly (soundness is checked against the exact golden certificate).
Reports, per (s, pb), the share of core-steps certified relative to the exact certificate and the datapath widths.
Usage: python lowprec.py <data_dir> <model> <samples>"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from golden import Net, F
from energy_estimate import load

d, m, NS = sys.argv[1], sys.argv[2], int(sys.argv[3]); K = 4; C = 32
net = load(d, m)
g = Net(net["W"], net["D"], net["I"], net["beta"], net["theta"], net["rho"], net["gamma"], C)
Bk = [np.broadcast_to(g.bounded(k), (g.H,)).astype(np.int64) for k in range(1, K + 1)]


def cdiv(x, s):  # ceil(x / 2^s)
    return -((-x) >> s)


runs = []
for b in range(NS):
    S, cert, viol, V, A, slots = g.run(b, K=K)
    runs.append((S, cert, V, A, slots, b))

grid = [(s, pb) for s in (0, 2, 4, 6, 8) for pb in (12, 8, 6, 4)]
for s, pb in grid:
    bu = -((-int(g.beta)) >> (F - pb)); bd = int(g.beta) >> (F - pb); rd = int(g.rho) >> (F - pb)
    th = int(g.theta) >> s; gm = int(g.gamma) >> s
    n_exact = n_low = n_bad = n_tot = 0; umax = 0
    for S, cert, V, A, slots, b in runs:
        T = S.shape[0]
        for t in range(1, T - K):
            U = cdiv(V[t], s); Al = A[t] >> s; fire = np.zeros(g.H, bool)
            for k in range(1, K + 1):
                x = cdiv(slots[t, k - 1] + g.I[b, t + k], s) + cdiv(Bk[k - 1], s)
                U = np.where(U >= 0, -((-(bu * U)) >> pb), -((-(bd * U)) >> pb)) + x
                Al = ((rd * Al) >> pb) + (gm * S[t] if k == 1 else 0)
                fire |= U >= th + Al
                umax = max(umax, int(np.abs(U).max()))
            low = ~fire.reshape(g.P, C).any(1)
            n_low += int(low.sum()); n_exact += int(cert[t].sum()); n_tot += g.P
            n_bad += int((low & ~cert[t]).sum())
    print(f"RESULT model={m} s={s} pb={pb} operand_bits={int(np.ceil(np.log2(umax + 1))) + 1} beta_bits={pb + 1} "
          f"certified_exact={n_exact / n_tot:.4f} certified_low={n_low / n_tot:.4f} retained={n_low / max(n_exact, 1):.3f} "
          f"unsound={n_bad}", flush=True)
