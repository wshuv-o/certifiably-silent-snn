"""HW-006: certificate-stage work with early abort. In a normal pass the core certificate fails at the first neuron
whose K-step bound reaches threshold; stages 2..K are then isolated for the remaining neurons of the pass. Replays the
hardware schedule (normal / idle / catch-up) per core from the golden model and reports, per core-step: fraction of
normal, catch-up and idle passes, and the mean fraction of a normal pass's neurons that use stages 2..K.
Usage: python early_abort.py <data_dir> <model> <samples> [K]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from golden import Net, mulc
from energy_estimate import load
d, m, NS = sys.argv[1], sys.argv[2], int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
net = load(d, m); C = 32
g = Net(net["W"], net["D"], net["I"], net["beta"], net["theta"], net["rho"], net["gamma"], C)
tot = dict(steps=0, normal=0, catch=0, idle=0, evaluated=0.0, passes_cert=0)
for b in range(NS):
    S, cert, viol, V, A, slots = g.run(b, K=K)
    T = S.shape[0]
    for p in range(g.P):
        skip = 0
        for tau in range(1, T):
            tot["steps"] += 1
            if skip == 0:
                tot["normal"] += 1
                en = tau >= 2 and tau - 1 + K < T
                if en:
                    t = tau - 1; cols = range(p * C, (p + 1) * C)
                    first = C
                    for j, i in enumerate(cols):                      # neuron order of the pass
                        uk, ak, f = V[t, i], A[t, i], False
                        for k in range(1, K + 1):
                            uk = mulc(g.beta, uk) + g.I[b, t + k, i] + slots[t, k - 1, i] + int(np.broadcast_to(g.bounded(k), (g.H,))[i])
                            ak = (mulc(g.rho, ak) + g.gamma * S[t, i]) if k == 1 else mulc(g.rho, ak)
                            f |= uk >= g.theta + ak
                        if f: first = j + 1; break
                    tot["evaluated"] += first / C
                    tot["passes_cert"] += 1
                    if cert[t, p]: skip = K - 1
            elif skip == 1:
                tot["catch"] += 1; skip = 0
            else:
                tot["idle"] += 1; skip -= 1
s = tot["steps"]
print(f"RESULT model={m} samples={NS} normal={tot['normal']/s:.3f} catchup={tot['catch']/s:.3f} idle={tot['idle']/s:.3f} "
      f"stage2plus_fraction_per_normal_pass={tot['evaluated']/max(tot['passes_cert'],1):.3f}")
