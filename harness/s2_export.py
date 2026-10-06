"""ENGINE-002: export dense 512-ALIF S2 models (seed 1, lambda 0 and 0.3) for the C++ engine."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("DATASET", "shd")
import s2_strong as s

OUT = os.path.expanduser("~/research/data/e2"); os.makedirs(OUT, exist_ok=True)
Xte, yte = s.fetch("test")
samples = np.random.default_rng(1).choice(len(yte), 200, replace=False)
X = Xte[samples].astype(np.float64)
for tag, path in (("control", "~/research/models/s2_l0_s1.pt"), ("certified", "~/research/models/s2_l0.3_s1.pt")):
    sd = torch.load(os.path.expanduser(path), map_location="cpu")
    Win = sd["win.weight"].numpy().astype(np.float64); W = sd["wrec.weight"].numpy().astype(np.float64)
    I = np.einsum("btn,hn->bth", X, Win)
    W.tofile(f"{OUT}/{tag}_W.bin"); I.tofile(f"{OUT}/{tag}_I.bin")
    print(tag, W.shape, I.shape, "R mean", np.maximum(W, 0).sum(1).mean())
with open(f"{OUT}/dims.txt", "w") as f:
    P = int(os.environ.get("CORES", "16")); f.write(f"{P} {512 // P} {s.T} {len(samples)} {s.BETA!r} {s.THETA!r} {s.RHO!r} {s.GAMMA!r}\n")
