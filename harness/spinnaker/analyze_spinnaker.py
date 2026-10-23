"""Compare the SpiNNaker run (spinnaker_result.npz) with float references of the same network.

References are run on the same continuous timeline as the hardware (50 samples, 30 ms gaps):
  lif_d   LIF, hard reset to 0, recurrent spike at t reaches v at t+d   (what was requested on SpiNNaker)
  lif_d1  as lif_d with one extra step of recurrent latency
  alif    the trained model (subtractive reset, threshold adaptation), run per sample
Reports spike agreement, readout accuracy, silence, and K-step certificates evaluated on the recorded
SpiNNaker membranes, with any certified core that spiked counted as a violation.
Usage: python analyze_spinnaker.py"""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = np.load(os.path.join(HERE, "spinnaker_pack.npz")); R = np.load(os.path.join(HERE, "spinnaker_result.npz"))
win, W, wout, X, y = P["win"].astype(np.float64), P["W"].astype(np.float64), P["wout"].astype(np.float64), P["X"], P["y"]
delays = [int(d) for d in P["delays"]]; T = int(P["T"]); beta = float(P["beta"]); theta = float(P["theta"])
rho, gamma, beta_out = float(P["rho"]), float(P["gamma"]), float(P["beta_out"])
NS, H = X.shape[0], W.shape[1]; C = 32; NC = H // C
v_hw = R["v"].astype(np.float64); starts = R["starts"]; TT = v_hw.shape[0]
RESET = int(R["reset"]) if "reset" in R.files else 1          # recorded spike time precedes the membrane crossing
tf = R["spk_t"] + RESET; keep = tf < TT
S_hw = np.zeros((TT, H), np.uint8); S_hw[tf[keep], R["spk_i"][keep]] = 1

I = np.zeros((TT, H))
for b in range(NS):
    I[starts[b]:starts[b] + T] = X[b].astype(np.float64) @ win.T


def lif(extra):
    v = np.zeros(H); S = np.zeros((TT, H), np.uint8); V = np.zeros((TT, H))
    for t in range(TT):
        rec = sum(W[k] @ S[t - d - extra] for k, d in enumerate(delays) if t - d - extra >= 0)
        v = beta * v + I[t] + rec
        s = v >= theta; v = np.where(s, 0.0, v); S[t] = s; V[t] = v
    return S, V


def alif(b):
    v = np.zeros(H); a = np.zeros(H); s = np.zeros(H); S = np.zeros((T, H))
    It = X[b].astype(np.float64) @ win.T
    for t in range(T):
        a = rho * a + gamma * s
        rec = sum(W[k] @ S[t - d] for k, d in enumerate(delays) if t - d >= 0)
        v = beta * v + It[t] + rec; thr = theta + a
        s = (v >= thr).astype(float); v = v - s * thr; S[t] = s
    return S


def readout(S):
    u = np.zeros(wout.shape[0]); out = 0
    for t in range(S.shape[0]):
        u = beta_out * u + wout @ S[t]; out = out + u
    return int(np.argmax(out))


def windows(S):
    return [S[starts[b]:starts[b] + T] for b in range(NS)]


def compare(name, S):
    inwin = np.zeros(TT, bool)
    for b in range(NS): inwin[starts[b]:starts[b] + T] = True
    a, r = S_hw[inwin].astype(bool), S[inwin].astype(bool)
    both = (a & r).sum(); out_gap = int(S_hw[~inwin].sum())
    print(f"{name:7s} spikes ref {r.sum():6d} hw {a.sum():6d}  exact-time match {both / max(r.sum(), 1):.3f} of ref, "
          f"{both / max(a.sum(), 1):.3f} of hw;  per-neuron-per-sample count corr "
          f"{np.corrcoef(np.ravel([w.sum(0) for w in windows(S)]), np.ravel([w.sum(0) for w in windows(S_hw)]))[0, 1]:.3f}"
          f";  hw spikes in gaps {out_gap}")


S_d, V_d = lif(0); S_d1, V_d1 = lif(1)
compare("lif_d", S_d); compare("lif_d1", S_d1)
print("membrane |v_hw - v_ref| median:  lif_d", np.median(np.abs(v_hw - V_d)).round(4), " lif_d1", np.median(np.abs(v_hw - V_d1)).round(4))

acc = {}
acc["spinnaker"] = np.mean([readout(w) == y[b] for b, w in enumerate(windows(S_hw))])
acc["lif_d"] = np.mean([readout(w) == y[b] for b, w in enumerate(windows(S_d))])
acc["lif_d1"] = np.mean([readout(w) == y[b] for b, w in enumerate(windows(S_d1))])
S_alif = [alif(b) for b in range(NS)]
acc["alif (trained)"] = np.mean([readout(S_alif[b]) == y[b] for b in range(NS)])
agree = np.mean([readout(w) == readout(S_alif[b]) for b, w in enumerate(windows(S_hw))])
print("readout accuracy on", NS, "samples:", {k: round(float(v), 3) for k, v in acc.items()}, " hw-vs-alif label agreement", round(float(agree), 3))
print("spike rate per neuron-step: hw", round(float(np.mean([w.mean() for w in windows(S_hw)])), 4),
      " alif", round(float(np.mean([s.mean() for s in S_alif])), 4))

# silence per core-step inside sample windows
cs = np.stack([w.reshape(T, NC, C).any(-1) for w in windows(S_hw)])          # (NS, T, NC) core fired
print("silent core-steps (hw):", round(float(1 - cs.mean()), 4))

# K-step certificates on recorded membranes, base threshold, may-fire set = all neurons
def certificates(Wc, winc, name):
    Ic = np.zeros((TT, H))
    for b in range(NS):
        Ic[starts[b]:starts[b] + T] = X[b].astype(np.float64) @ winc.T
    Wp = np.maximum(Wc, 0)
    for K in (2, 4):
        short = sum(Wp[k].sum(1) for k, d in enumerate(delays) if d < K)          # bound of uncertain drive per step
        n_cert = n_viol = n_silent = 0
        for b in range(NS):
            t0 = starts[b]
            for t in range(t0, t0 + T - K):
                U = v_hw[t].copy(); ok = np.ones(H, bool)
                for k in range(1, K + 1):
                    tk = t + k
                    known = sum(Wc[i] @ S_hw[tk - d] for i, d in enumerate(delays) if d >= k)
                    U = beta * U + Ic[tk] + known + (short if any(d < k for d in delays) else 0)
                    ok &= U < theta
                core_ok = ok.reshape(NC, C).all(1)
                fired = S_hw[t + 1:t + K + 1].reshape(K, NC, C).any((0, 2))
                n_cert += core_ok.sum(); n_viol += (core_ok & fired).sum(); n_silent += (~fired).sum()
        print(f"{name} K={K}: certified core-windows {n_cert} of {n_silent} silent ({n_cert / max(n_silent, 1):.3f}), violations {n_viol}")


certificates(W, win, "nominal weights")
WF = os.path.join(HERE, "spinnaker_weights.npz")
Wsrc = R if "W_hw" in R.files else (np.load(WF) if os.path.exists(WF) else None)
if Wsrc is not None:
    W_hw, win_hw = Wsrc["W_hw"].astype(np.float64), Wsrc["win_hw"].astype(np.float64)
    print("stored vs nominal weights: max abs difference input", np.abs(win_hw - win).max().round(5), "recurrent", np.abs(W_hw - W).max().round(5),
          " total |w| ratio input", (np.abs(win_hw).sum() / np.abs(win).sum()).round(4), "recurrent", (np.abs(W_hw).sum() / np.abs(W).sum()).round(4))
    certificates(W_hw, win_hw, "chip-stored weights")
