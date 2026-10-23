"""Silent-run statistics per network: for each core (16 x 32 neurons), the lengths of maximal runs of consecutive
steps in which no neuron of the core spikes, from the exact engine dynamics on the exported test samples.
Reports the fraction of silent core-steps, the mean run length, and the share of silent core-steps that lie in
runs of at least K+1 steps (the runs a K-step certificate can exploit).
Usage: python silent_runs.py <data_dir> <model> [samples]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_estimate import load, simulate

d, m = sys.argv[1], sys.argv[2]; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
net = load(d, m); H = net["H"]; C = 32; P = H // C
runs = []; silent = total = 0
for b in range(min(NS, net["N"])):
    _, S, _ = simulate(net, b)
    core_spk = S[1:].reshape(S.shape[0] - 1, P, C).sum(-1) > 0          # (T-1, P)
    for p in range(P):
        r = 0
        for t in range(core_spk.shape[0]):
            if not core_spk[t, p]:
                r += 1
            else:
                if r: runs.append(r)
                r = 0
        if r: runs.append(r)
    silent += int((~core_spk).sum()); total += core_spk.size
runs = np.array(runs)
long_share = {K: runs[runs >= K + 1].sum() / runs.sum() for K in (4, 8)}
print(f"RESULT model={m} samples={min(NS, net['N'])} silent_frac={silent / total:.3f} mean_run={runs.mean():.2f} "
      f"median_run={np.median(runs):.0f} share_in_runs_ge5={long_share[4]:.3f} share_in_runs_ge9={long_share[8]:.3f}")
