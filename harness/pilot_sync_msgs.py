"""PILOT-006 (pre-registered in research/BRAINSTORM_003_NEUROMORPHIC.md): synchronization
messages per core under lock-step vs LOCAL certified-silence protocol vs oracle.
Uses the saved PILOT-005 models (8 cores x 32 neurons, ring-local recurrence).
"""
import os, json
os.environ["LOCAL"] = "1"                       # must be set before importing the model module
import numpy as np
import torch
import pilot_silence as ps

KMAX = 16


@torch.no_grad()
def core_horizons(m, X):
    """c[b,t,p]: largest k<=KMAX s.t. every neuron of core p provably cannot reach theta
    within k steps after t (local, sound: worst-case recurrent drive from all presynaptic)."""
    W = (m.wrec.weight * m.mask).numpy()
    R = np.maximum(W, 0).sum(1)                                   # [H]
    _, iext, V, S, _ = m(torch.tensor(X, dtype=torch.float32))
    iext, V, S = iext.numpy(), V.numpy(), S.numpy()
    B, T, H = V.shape
    tmax = T - 1
    rec1 = np.einsum("btk,ik->bti", S[:, :tmax], W)               # exact input at t+1
    first = np.full((B, tmax, H), KMAX + 1, np.int16)             # first k at which bound may cross
    v = V[:, :tmax].copy()
    for k in range(1, KMAX + 1):
        idx = np.minimum(np.arange(tmax) + k, T - 1)              # clamp beyond the end
        v = ps.BETA * v + iext[:, idx] + (rec1 if k == 1 else R)
        hit = (v >= ps.THETA) & (first > KMAX)
        first[hit] = k
    neuron_h = first - 1                                          # silent steps guaranteed
    core_h = neuron_h.reshape(B, tmax, ps.NP, H // ps.NP).min(-1)  # [B,tmax,NP]
    core_spk = S.reshape(B, T, ps.NP, H // ps.NP).max(-1) > 0      # [B,T,NP]
    return core_h, core_spk


def count_messages(core_h, core_spk):
    B, tmax, NP = core_h.shape
    T = tmax + 1
    lock = cert = orac = 0
    for b in range(B):
        for p in range(NP):
            lock += T
            t = 0
            while t < T:                                          # certified protocol
                if core_spk[b, t, p] or t >= tmax or core_h[b, t, p] < 1:
                    cert += 1; t += 1
                else:
                    cert += 1; t += 1 + int(core_h[b, t, p])
            s = core_spk[b, :, p]                                 # oracle
            runs = np.sum(~s[1:] & s[:-1]) + (not s[0])           # starts of silent runs
            orac += int(s.sum()) + int(runs)
    return lock, cert, orac


def main():
    Xte, yte = ps.fetch("test")
    sub = np.random.default_rng(1).choice(len(yte), 500, replace=False)
    rows = []
    # PAIRS env: "label=ctrl.pt,cert.pt;..." (PILOT-007); default: PILOT-005 models
    pairs = [(p.split("=")[0], *p.split("=")[1].split(",")) for p in os.environ["PAIRS"].split(";")] \
        if os.environ.get("PAIRS") else [(f"s{s}", f"~/research/models/p005_s{s}_l0.pt",
                                            f"~/research/models/p005_s{s}_l0.1.pt") for s in (1, 2, 3)]
    for label, ctrl, cert_path in pairs:
        for lam, path in (("0", ctrl), ("0.1", cert_path)):
            seed = label
            m = ps.RSNN()
            m.load_state_dict(torch.load(os.path.expanduser(path)))
            m.eval()
            h, spk = core_horizons(m, Xte[sub])
            lock, cert, orac = count_messages(h, spk)
            rows.append(dict(seed=seed, lam=float(lam), lock=lock, cert=cert, oracle=orac,
                             reduction=lock / cert, oracle_reduction=lock / orac,
                             mean_horizon_when_silent=float(h[h >= 1].mean()) if (h >= 1).any() else 0.0,
                             frac_steps_certified=float((h >= 1).mean())))
            print(rows[-1], flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n", df.groupby(["seed", "lam"] if os.environ.get("PAIRS") else "lam")[["reduction", "oracle_reduction", "mean_horizon_when_silent",
                                   "frac_steps_certified"]].agg(["mean", "std"]).round(3).to_string())
    json.dump(rows, open(os.environ.get("OUT", "../results/pilot006.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
