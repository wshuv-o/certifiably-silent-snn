"""S1b: export ring-local model weights and external inputs as raw float64 for the C++ engine."""
import os, sys
os.environ["LOCAL"] = "1"
import numpy as np
import torch
import pilot_silence as ps

P, C, H = 8, 32, ps.H
OUT = os.path.expanduser("~/research/data/s1b")
os.makedirs(OUT, exist_ok=True)
Xte, yte = ps.fetch("test")
samples = np.random.default_rng(1).choice(len(yte), 200, replace=False)
X = Xte[samples].astype(np.float64)
MODELS = os.environ.get("MODELS")       # "tag=path,tag=path" (ENGINE-ALT); default: S1b/S1c pair
pairs = [m.split("=") for m in MODELS.split(",")] if MODELS else \
    [("control", "~/research/models/p005_s1_l0.pt"), ("certified", "~/research/models/p005_s1_l0.1.pt")]
for tag, path in pairs:
    m = ps.RSNN(); m.load_state_dict(torch.load(os.path.expanduser(path))); m.eval()
    W = (m.wrec.weight * m.mask).detach().numpy().astype(np.float64)
    Win = m.win.weight.detach().numpy().astype(np.float64)
    I = np.einsum("btn,hn->bth", X, Win)                                   # [n,T,H]
    nbr = [((p - 1) % P, p, (p + 1) % P) for p in range(P)]
    Wloc = np.stack([np.concatenate([W[p*C:(p+1)*C, q*C:(q+1)*C] for q in nbr[p]], 1) for p in range(P)])  # [P,C,3C]
    Wloc.tofile(f"{OUT}/{tag}_Wloc.bin"); I.tofile(f"{OUT}/{tag}_I.bin")
    print(tag, Wloc.shape, I.shape, "R mean", np.maximum(Wloc, 0).sum(2).mean())
with open(f"{OUT}/dims.txt", "w") as f:
    f.write(f"{P} {C} {ps.T} {len(samples)} {ps.BETA!r} {ps.THETA!r}\n")
