"""Package the locally connected K=4 network (sp5_lk512) and 50 SHD test samples for SpiNNaker.

Writes spinnaker_pack.npz with input weights, recurrent taps (masked), readout weights, binned input counts,
labels and model constants, and checks the package against the engine export (lk4_* in data/e6).
Usage: python pack_spinnaker.py [n_samples]"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import pilot_silence as ps

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
M = os.path.expanduser("~/research/models/sp5_lk512.pt"); E = os.path.expanduser("~/research/data/e6")
sd = torch.load(M, map_location="cpu")
mask = sd["mask"].numpy().astype(np.float64)
W = [sd[f"wrec.{i}"].numpy().astype(np.float64) * mask for i in range(3)]
win = sd["win.weight"].numpy().astype(np.float64); wout = sd["wout.weight"].numpy().astype(np.float64)
H = W[0].shape[0]

# check against the engine export
for i in range(3):
    We = np.fromfile(f"{E}/lk4_W{i}.bin", dtype=np.float64).reshape(H, H)
    assert np.allclose(We, W[i]), f"tap {i} differs from lk4 export"
Xte, yte = ps.fetch("test")
idx = np.random.default_rng(1).choice(len(yte), 200, replace=False)[:NS]       # same samples as the export
X = Xte[idx].astype(np.float64); y = yte[idx]
I_pack = np.einsum("btn,hn->bth", X, win)
I_exp = np.fromfile(f"{E}/lk4_I.bin", dtype=np.float64).reshape(200, 100, H)[:NS]
assert np.allclose(I_pack, I_exp), "input current differs from export"

counts = X.astype(np.int64)
print("input counts per (step, channel): max", counts.max(), "share of nonzero bins with count>1",
      round(float((counts > 1).sum() / max((counts > 0).sum(), 1)), 4),
      "share of input mass in counts>3", round(float(np.maximum(counts - 3, 0).sum() / counts.sum()), 5))
print("recurrent synapses per tap", [int((w != 0).sum()) for w in W], "input synapses", int((win != 0).sum()))
BETA_OUT = float(np.exp(-0.5))                               # readout decay of s5_delays.py
np.savez_compressed(os.path.join(HERE, "spinnaker_pack.npz"), win=win.astype(np.float32),
                    W=np.stack(W).astype(np.float32), wout=wout.astype(np.float32), X=counts.astype(np.uint8),
                    y=y.astype(np.int16), delays=np.array([2, 4, 8]), beta=np.exp(-0.5), theta=1.0,
                    rho=np.exp(-14 / 200), gamma=0.02, beta_out=BETA_OUT, T=100)
print("saved", os.path.join(HERE, "spinnaker_pack.npz"), "samples", NS)
