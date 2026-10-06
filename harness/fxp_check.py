"""FXP-001 (pre-registered in research/N3_SCALEUP_PLAN.md): are certificates sound under chip-style
fixed-point arithmetic?

Quantization (Loihi-style CUBA-like integer neuron):
  q = max(|W_in|, |W_rec|) / 127            shared scale, so inputs and recurrent drive share units
  W_int = round(W / q) in [-127, 127];  theta_int = round(theta / q)
  v <- v - floor(v * d / 4096) + I_int + W_rec_int @ s;   d = round((1 - beta) * 4096)
  spike if v >= theta_int, reset by subtracting theta_int.
The decay map f(v) = v - floor(v*d/4096) is monotone non-decreasing (d < 4096), so Lemma 1/2 carry over
exactly in integer arithmetic with R_int = sum of positive W_rec_int.
"""
import os, json
os.environ["LOCAL"] = "1"
import numpy as np
import torch
import pilot_silence as ps

K = 4


def quantize(m):
    Win = m.win.weight.detach().numpy().astype(np.float64)
    W = (m.wrec.weight * m.mask).detach().numpy().astype(np.float64)
    q = max(np.abs(Win).max(), np.abs(W).max()) / 127.0
    return (np.round(Win / q).astype(np.int64), np.round(W / q).astype(np.int64),
            int(round(ps.THETA / q)), int(round((1 - ps.BETA) * 4096)), q)


def decay(v, d):
    return v - np.floor_divide(v * d, 4096)


def simulate(Win_i, W_i, th, d, X):
    B = X.shape[0]; H = W_i.shape[0]
    Iext = np.einsum("btn,hn->bth", X.astype(np.int64), Win_i)        # exact integers
    v = np.zeros((B, H), np.int64); s = np.zeros((B, H), np.int64)
    V = np.zeros((B, ps.T, H), np.int64); S = np.zeros((B, ps.T, H), np.int64)
    for t in range(ps.T):
        v = decay(v, d) + Iext[:, t] + s @ W_i.T
        s = (v >= th).astype(np.int64)
        v = v - s * th
        V[:, t] = v; S[:, t] = s
    return Iext, V, S


def certificates(Iext, V, S, W_i, th, d):
    B, T, H = V.shape; tmax = T - K
    Wp = np.maximum(W_i, 0)
    rec1 = np.einsum("btk,ik->bti", S[:, :tmax], W_i)
    def may_fire(R):
        v = V[:, :tmax].copy(); fire = np.zeros_like(v, bool)
        for k in range(1, K + 1):
            v = decay(v, d) + Iext[:, k:tmax + k] + (rec1 if k == 1 else R); fire |= v >= th
        return fire
    Sset = np.ones((B, tmax, H), bool)
    for _ in range(50):
        new = may_fire(np.einsum("btk,ik->bti", Sset.astype(np.int64), Wp)) & Sset
        if np.array_equal(new, Sset): break
        Sset = new
    actual = np.zeros((B, tmax, H), bool)
    for k in range(1, K + 1): actual |= S[:, k:tmax + k] > 0
    part = (~Sset).reshape(B, tmax, ps.NP, H // ps.NP).all(-1)
    nbr = np.roll(part, 1, -1) & np.roll(part, -1, -1)
    apart = (~actual).reshape(B, tmax, ps.NP, H // ps.NP).all(-1)
    anbr = np.roll(apart, 1, -1) & np.roll(apart, -1, -1)
    return dict(violations=int((~Sset & actual).sum()), certified_windows=int((~Sset).sum()),
                nbr_cert=float(nbr.mean()), nbr_oracle=float(anbr.mean()))


def readout_acc(m, S, y):
    Wout = m.wout.weight.detach().numpy()
    u = np.zeros((S.shape[0], Wout.shape[0])); out = 0
    for t in range(ps.T):
        u = ps.BETA * u + S[:, t].astype(np.float64) @ Wout.T; out = out + u
    return float((out.argmax(1) == y).mean())


def main():
    Xte, yte = ps.fetch("test")
    rows = []
    PATTERN = os.environ.get("PATTERN", "~/research/models/p005_s{}_l0.1.pt")   # FXP-ExCap: alt_clamp_s{}.pt
    for seed in (1, 2, 3):
        m = ps.RSNN(); m.load_state_dict(torch.load(os.path.expanduser(PATTERN.format(seed))))
        Win_i, W_i, th, d, q = quantize(m)
        accs, agg = [], dict(violations=0, certified_windows=0, nbr_cert=0.0, nbr_oracle=0.0); n = 0
        for i in range(0, len(yte), 250):
            Iext, V, S = simulate(Win_i, W_i, th, d, Xte[i:i + 250])
            accs.append(readout_acc(m, S, yte[i:i + 250]) * len(S))
            c = certificates(Iext, V, S, W_i, th, d)
            agg["violations"] += c["violations"]; agg["certified_windows"] += c["certified_windows"]
            agg["nbr_cert"] += c["nbr_cert"] * len(S); agg["nbr_oracle"] += c["nbr_oracle"] * len(S); n += len(S)
        row = dict(seed=seed, theta_int=th, decay_d=d, q=q, int_accuracy=sum(accs) / n,
                   violations=agg["violations"], certified_windows=agg["certified_windows"],
                   nbr_cert=agg["nbr_cert"] / n, nbr_oracle=agg["nbr_oracle"] / n)
        rows.append(row); print(row, flush=True)
    json.dump(rows, open(os.environ.get("OUTJSON", "../results/fxp001.json"), "w"), indent=1)
    print("TOTAL violations:", sum(r["violations"] for r in rows),
          "| certified neuron-windows checked:", sum(r["certified_windows"] for r in rows))


if __name__ == "__main__":
    main()
