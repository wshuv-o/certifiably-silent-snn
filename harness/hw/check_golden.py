"""Compare the fixed-point golden model with the float engine dynamics; report spike agreement, the share of
(core, step) pairs certified by the hardware certificate, and violations."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from golden import Net
from energy_estimate import load, simulate
d, m = sys.argv[1], sys.argv[2]; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 5
net = load(d, m)
g = Net(net["W"], net["D"], net["I"], net["beta"], net["theta"], net["rho"], net["gamma"], net["C"])
agree = tot = viol = cert = cs = 0; sp_fx = sp_fl = 0
for b in range(nb):
    S, c, v, *_ = g.run(b, K=4, adapt=True)
    _, Sf, _ = simulate(net, b)
    agree += int((S == Sf).sum()); tot += S.size; viol += v; cert += int(c.sum()); cs += c.size
    sp_fx += int(S.sum()); sp_fl += int(Sf.sum())
print(f"model={m} samples={nb} spike_agreement={agree/tot:.5f} spikes_fx={sp_fx} spikes_float={sp_fl} "
      f"certified_frac={cert/cs:.3f} violations={viol}")
