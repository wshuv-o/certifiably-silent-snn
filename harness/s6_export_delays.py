"""Export an s5_delays.py model for the delay-aware C++ engine (s6_engine_delays.cpp).

Usage: MODELS="tag=path,..." CORES=32 python s6_export_delays.py

Writes per delay tap {tag}_W{idx}.bin (H x H, masked), plus {tag}_I.bin (n x T x H) and
{tag}_dims.txt = "P C T N BETA THETA RHO GAMMA NDELAY d1 d2 ...".

The engine needs one weight matrix per tap because the recurrent drive at step t+1 is
sum_d W_d s(t+1-d); the single-delay exporter (s3_export.py) cannot represent that.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_silence as ps

OUT = os.path.expanduser("~/research/data/e6"); os.makedirs(OUT, exist_ok=True)
CORES = int(os.environ.get("CORES", "16"))
NSAMP = int(os.environ.get("NSAMP", "200"))
TAUM = float(os.environ.get("TAUM", "2.0"))
BETA = float(np.exp(-1.0 / TAUM)); RHO, GAMMA = float(np.exp(-14 / 200)), 0.02

Xte, yte = ps.fetch("test")
samples = np.random.default_rng(1).choice(len(yte), NSAMP, replace=False)
X = Xte[samples].astype(np.float64)

for item in os.environ["MODELS"].split(","):
    tag, path = item.split("=", 1)
    sd = torch.load(os.path.expanduser(path), map_location="cpu")
    # s5_delays stores taps as wrec.0, wrec.1, ... (a ParameterList)
    keys = sorted([k for k in sd if k.startswith("wrec.")], key=lambda k: int(k.split(".")[1]))
    assert keys, f"no wrec.* taps found in {path}; is this an s5_delays checkpoint?"
    mask = sd["mask"].numpy().astype(np.float64) if "mask" in sd else None
    Ws = []
    for k in keys:
        Wd = sd[k].numpy().astype(np.float64)
        if mask is not None:
            Wd = Wd * mask
        Ws.append(Wd)
    H = Ws[0].shape[0]; P = CORES; C = H // P
    assert H % P == 0, f"H={H} not divisible by CORES={P}"
    Win = sd["win.weight"].numpy().astype(np.float64)
    # This recomputes the feedforward current instead of calling the model, so any transform the
    # model applies to it must be replicated here. ACC-001 can add input binning and feedforward
    # batch normalisation; neither is implemented below, so refuse rather than export a current the
    # network never saw.
    assert "traw" not in sd, (
        "checkpoint was trained with per-neuron learnable time constants (TAU_LEARN=1); dims.txt "
        "carries a single scalar beta, so the engine would simulate different dynamics than the "
        "network was trained with. Extend dims.txt to a beta vector before exporting.")
    assert not any(k.startswith("bn.") for k in sd), (
        "checkpoint was trained with feedforward batch normalisation (BN=1); the export does not "
        "replicate it. Fold the eval-mode affine into win.weight before exporting.")
    assert Win.shape[1] == X.shape[2], (
        f"win expects {Win.shape[1]} inputs but the data has {X.shape[2]}; the checkpoint was "
        f"trained with input binning (NBINS={X.shape[2] // Win.shape[1]}), which the export does "
        f"not replicate.")
    I = np.einsum("btn,hn->bth", X, Win)
    delays = [int(d) for d in os.environ.get("DELAYS", "2,4,8").split(",")]
    assert len(delays) == len(Ws), f"DELAYS has {len(delays)} entries but checkpoint has {len(Ws)} taps"
    for di, Wd in enumerate(Ws):
        Wd.tofile(f"{OUT}/{tag}_W{di}.bin")
    I.tofile(f"{OUT}/{tag}_I.bin")
    with open(f"{OUT}/{tag}_dims.txt", "w") as f:
        f.write(f"{P} {C} {ps.T} {len(samples)} {BETA!r} {ps.THETA!r} {RHO!r} {GAMMA!r} "
                f"{len(delays)} {' '.join(str(d) for d in delays)}\n")
    Rd = [float(np.maximum(Wd, 0).sum(1).mean()) for Wd in Ws]
    K = 4
    print(f"{tag}: H={H} P={P} C={C} delays={delays} R_per_delay={[round(r,3) for r in Rd]} "
          f"R_total={sum(Rd):.3f} R_short(d<{K})={sum(r for d,r in zip(delays,Rd) if d<K):.3f} "
          f"budget={(1-BETA)*ps.THETA:.4f}", flush=True)
