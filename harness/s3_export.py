"""SWEEP-001 / B: export any trained model for the general C++ engine (s3_engine.cpp).
Usage: MODELS="tag=kind:path,..." python s3_export.py
  kind = lif_local (pilot_silence RSNN, 8x32 ring mask, beta=e^-0.5, no adaptation)
       | alif      (s2_strong / s3_sweep ALIFNet, dense or hub-masked, 16x32 default)
Writes {tag}_W.bin (H x H, masked), {tag}_I.bin (n x T x H), {tag}_dims.txt.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("LOCAL", "1")
import pilot_silence as ps

OUT = os.path.expanduser("~/research/data/e3"); os.makedirs(OUT, exist_ok=True)
Xte, yte = ps.fetch("test")
samples = np.random.default_rng(1).choice(len(yte), 200, replace=False)
X = Xte[samples].astype(np.float64)
CORES = int(os.environ.get("CORES", "8"))
for item in os.environ["MODELS"].split(","):
    tag, rest = item.split("="); kind, path = rest.split(":", 1)
    sd = torch.load(os.path.expanduser(path), map_location="cpu")
    W = sd["wrec.weight"].numpy().astype(np.float64)
    if "mask" in sd:
        W = W * sd["mask"].numpy().astype(np.float64)
    Win = sd["win.weight"].numpy().astype(np.float64)
    H = W.shape[0]; P = CORES; C = H // P
    I = np.einsum("btn,hn->bth", X, Win)
    if kind == "lif_local":
        beta, rho, gamma = float(np.exp(-0.5)), 0.0, 0.0
    else:
        beta, rho, gamma = float(np.exp(-0.5)), float(np.exp(-14 / 200)), 0.02
    W.tofile(f"{OUT}/{tag}_W.bin"); I.tofile(f"{OUT}/{tag}_I.bin")
    with open(f"{OUT}/{tag}_dims.txt", "w") as f:
        f.write(f"{P} {C} {ps.T} {len(samples)} {beta!r} {ps.THETA!r} {rho!r} {gamma!r}\n")
    print(tag, kind, "H", H, "P", P, "C", C, "R mean", np.maximum(W, 0).sum(1).mean(), flush=True)
