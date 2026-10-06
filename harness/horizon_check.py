"""HORIZON-001 (pre-registered in research/N3_SCALEUP_PLAN.md): test the Proposition.
For each neuron: margin m_i = (1-beta)*theta - R_i - a_i, with a_i = 90th percentile of its external
input over the test set (a "typical quiet input" level). Lemma-1 certified horizon (worst-case R from the
first future step, as in the engine), capped at 64 steps and at the end of the sample, computed at every
step where the neuron is silent.
Prediction: median horizon(m_i > 0) >= 4x median horizon(m_i <= 0); falsified if < 2x.
"""
import os, json
os.environ["LOCAL"] = "1"
import numpy as np
import torch
from scipy.stats import spearmanr
import pilot_silence as ps

KMAX = 64


@torch.no_grad()
def main():
    Xte, yte = ps.fetch("test")
    sub = np.random.default_rng(1).choice(len(yte), 300, replace=False)
    out = []
    for seed in (1, 2, 3):
        m = ps.RSNN(); m.load_state_dict(torch.load(os.path.expanduser(f"~/research/models/p005_s{seed}_l0.1.pt")))
        W = (m.wrec.weight * m.mask).numpy(); R = np.maximum(W, 0).sum(1)
        _, iext, V, S, _ = m(torch.tensor(Xte[sub], dtype=torch.float32))
        iext, V, S = iext.numpy(), V.numpy(), S.numpy()
        B, T, H = V.shape
        a = np.percentile(iext.reshape(-1, H), 90, axis=0)
        margin = (1 - ps.BETA) * ps.THETA - R - a
        # first future step k at which the worst-case bound may cross theta (inputs beyond the end = 0)
        first = np.full((B, T, H), KMAX + 1, np.int32)
        u = V.copy()
        for k in range(1, KMAX + 1):
            idx = np.arange(T) + k
            inp = np.where((idx < T)[None, :, None], iext[:, np.minimum(idx, T - 1)], 0.0)
            u = ps.BETA * u + inp + R
            hit = (u >= ps.THETA) & (first > KMAX)
            first[hit] = k
        remaining = (T - 1 - np.arange(T))[None, :, None]
        hor = np.minimum(np.minimum(first - 1, KMAX), remaining)
        silent = S == 0
        hs = np.where(silent, hor, np.nan).reshape(-1, H)
        mean_h = np.nanmean(hs, 0)
        pos, neg = margin > 0, margin <= 0
        med = lambda mask: float(np.nanmedian(hs[:, mask])) if mask.any() else None
        row = dict(seed=seed, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                   median_h_pos=med(pos), median_h_neg=med(neg),
                   mean_h_pos=float(np.nanmean(hs[:, pos])) if pos.any() else None,
                   mean_h_neg=float(np.nanmean(hs[:, neg])) if neg.any() else None,
                   spearman_margin_meanh=float(spearmanr(margin, mean_h).correlation),
                   R_mean=float(R.mean()), a_mean=float(a.mean()))
        out.append(row); print(row, flush=True)
    json.dump(out, open("../results/horizon001.json", "w"), indent=1)


if __name__ == "__main__":
    main()
