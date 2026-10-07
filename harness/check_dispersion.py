"""Does the spread of R_i across neurons explain what its mean does not?

The governing relation holds cleanly at the extremes: below R_short ~0.4 everything certifies 95-98%
of oracle, above ~4.8 nothing does. In between it inverts -- local connectivity at R_short 1.221
certifies 34.6% while dense at 1.611 certifies 97.0%.

Hypothesis: a core certifies only if EVERY neuron in it stays below threshold, so the binding
statistic is not the mean of R_i but the per-core maximum. R_i sums over fan-in, so a 96-input local
neuron has a more variable R_i than a 512-input dense one at the same mean, and more cores are
spoiled by a single high-R_i neuron.

Prediction: the fraction of cores whose maximum R_i is small tracks the certified fraction better
than the mean R_short does.

    python harness/check_dispersion.py
"""
import os

import numpy as np
import torch

M = os.path.expanduser("~/research/models")
CPC = 32                 # neurons per core
K = 4                    # certificate horizon; taps with d < K bind
BUDGET = float((1.0 - np.exp(-0.5)) * 1.0)

# checkpoint, label, delay set, measured certified fraction (% of oracle) from the logs
CASES = [
    ("d2_E3_mindelay2_long.pt", "dense  {2,4,8} ctrl",  [2, 4, 8],  6.3),
    ("d3_F2_cert10.pt",         "dense  {2,4,8} cons",  [2, 4, 8], 94.9),
    ("dc_DC1_ctrl.pt",          "dense  learnable ctrl", None,     97.4),
    ("dc_DC1_cert.pt",          "dense  learnable cons", None,     98.0),
    ("sl_lc512.pt",             "local  H=512  ctrl",   [2, 4, 8], 34.6),
    ("sl_lc1024.pt",            "local  H=1024 ctrl",   [2, 4, 8], 84.9),
    ("sl_lc2048.pt",            "local  H=2048 ctrl",   [2, 4, 8], 90.0),
    ("sl_lk512.pt",             "local  H=512  cons",   [2, 4, 8], 97.4),
    ("sl_lk1024.pt",            "local  H=1024 cons",   [2, 4, 8], 96.3),
    ("sl_lk2048.pt",            "local  H=2048 cons",   [2, 4, 8], 98.5),
]


def r_short_per_neuron(path, delays):
    """Worst-case excitatory drive into each neuron through synapses with delay < K."""
    sd = torch.load(os.path.join(M, path), map_location="cpu")
    if delays is None:                       # per-synapse learnable delays
        w = sd["w"].float()
        D = 2.0 + 6.0 * torch.sigmoid(sd["draw"].float())
        tot = torch.zeros(w.shape[0])
        for k in range(1, K):                # taps k = 1..K-1 bind
            Wk = w * torch.relu(1.0 - (D - k).abs())
            tot += torch.clamp(Wk, min=0).sum(1)
        return tot.numpy()
    mask = sd.get("mask")
    tot = torch.zeros(sd["wrec.0"].shape[0])
    for i, d in enumerate(delays):
        if d >= K:
            continue
        W = sd["wrec.%d" % i].float()
        if mask is not None:
            W = W * mask.float()
        tot += torch.clamp(W, min=0).sum(1)
    return tot.numpy()


print("%-22s %6s %7s %7s %7s %8s %8s %7s" % (
    "model", "H", "mean", "sd", "cv", "coremax", "cores_ok", "cert%"))
print("-" * 86)
rows = []
for path, label, delays, cert in CASES:
    if not os.path.exists(os.path.join(M, path)):
        print("%-22s  MISSING" % label)
        continue
    r = r_short_per_neuron(path, delays)
    H = len(r)
    cores = r.reshape(-1, CPC)
    cmax = cores.max(1)                       # the binding neuron in each core
    # a core can only certify if its worst neuron is tolerable; the scale is set by the budget
    ok = float((cmax < 4.0 * BUDGET).mean()) * 100
    cv = r.std() / r.mean() if r.mean() > 0 else 0.0
    print("%-22s %6d %7.3f %7.3f %7.3f %8.3f %7.1f%% %6.1f" % (
        label, H, r.mean(), r.std(), cv, cmax.mean(), ok, cert))
    rows.append((label, r.mean(), cv, cmax.mean(), ok, cert))

if len(rows) > 2:
    import statistics as st

    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        va = sum((x - ma) ** 2 for x in a) ** 0.5
        vb = sum((x - mb) ** 2 for x in b) ** 0.5
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) if va * vb else 0.0

    cert = [r[5] for r in rows]
    print("\ncorrelation with certified fraction, across %d models:" % len(rows))
    print("  mean R_short          %+.3f" % corr([r[1] for r in rows], cert))
    print("  coefficient of var.   %+.3f" % corr([r[2] for r in rows], cert))
    print("  mean per-core max     %+.3f" % corr([r[3] for r in rows], cert))
    print("  %% cores under bound   %+.3f" % corr([r[4] for r in rows], cert))
    print("\nIf the last line is clearly stronger than the first, dispersion is the missing term.")
