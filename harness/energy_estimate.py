"""ENERGY-001: energy that silence certificates make avoidable on a time-stepped neuromorphic core.

On Loihi every neuron of a core is updated at every algorithmic time step, at 52 pJ when inactive and 81 pJ
when active (Davies et al. 2018 IEEE Micro 38 82, table 2, pre-silicon at 0.75 V). While a core is certified
silent over a window it cannot spike, so its membrane evolves linearly (decay plus accumulated input, no reset)
and its per-step neuron updates can be replaced by one closed-form catch-up update at the end of the window;
synaptic events are still processed (accumulated into the catch-up). This script replays each exported network
exactly (the engine's dynamics, float64), computes the per-core certificate with the bound of the paper (base or
adaptation-aware), walks each core greedily through certified windows, and prices:
  neuron updates   81 pJ if the neuron received input or spiked that step, else 52 pJ
  synaptic events  23.6 pJ per (spike, postsynaptic target) pair
  spike routing    1.7 pJ within tile + 3.5 pJ per hop, one message per destination core (mean hop count
                   of a square mesh of P cores)
  catch-up         81 pJ per neuron per certified window (conservative: priced as an active update)
Reports the avoidable fraction of neuron-update energy and of total energy, per network.
Usage: python energy_estimate.py <data_dir> <model> [K] [adapt]"""
import sys, numpy as np

E_NU_ACT, E_NU_INACT, E_SYN, E_SPK_TILE, E_HOP = 81e-12, 52e-12, 23.6e-12, 1.7e-12, 3.5e-12


def load(d, m):
    f = open(f"{d}/{m}_dims.txt").read().split()
    P, C, T, N = map(int, f[:4]); beta, theta, rho, gamma = map(float, f[4:8]); ND = int(f[8])
    D = [int(x) for x in f[9:9 + ND]]; H = P * C
    W = [np.fromfile(f"{d}/{m}_W{i}.bin", dtype=np.float64).reshape(H, H) for i in range(ND)]
    I = np.fromfile(f"{d}/{m}_I.bin", dtype=np.float64).reshape(N, T, H)
    return dict(P=P, C=C, T=T, N=N, beta=beta, theta=theta, rho=rho, gamma=gamma, D=D, W=W, I=I, H=H)


def simulate(net, b):
    """Engine dynamics for sample b: returns post-reset V, spikes S, adaptation A (T, H)."""
    T, H, D, W = net["T"], net["H"], net["D"], net["W"]
    v = np.zeros(H); a = np.zeros(H); S = np.zeros((T, H)); V = np.zeros((T, H)); A = np.zeros((T, H))
    for t in range(1, T):
        acc = np.zeros(H)
        for w, d in zip(W, D):
            if t - d >= 0:
                acc += w @ S[t - d]
        a = net["rho"] * a + net["gamma"] * S[t - 1]
        u = net["beta"] * v + net["I"][b, t] + acc
        thr = net["theta"] + a
        s = (u >= thr).astype(float)
        v = u - s * thr
        S[t], V[t], A[t] = s, v, a
    return V, S, A


def certified_cores(net, V, S, A, K, adapt):
    """certified[t, p] = core p provably silent at steps t+1..t+K (fixed-point bound of the paper)."""
    T, H, P, C, D, W = net["T"], net["H"], net["P"], net["C"], net["D"], net["W"]
    Wp = [np.maximum(w, 0) for w in W]; beta, theta, rho, gamma = net["beta"], net["theta"], net["rho"], net["gamma"]
    tmax = T - K
    Spad = np.vstack([np.zeros((max(D), H)), S])
    Sset = np.ones((tmax, H), bool)
    for _ in range(50):
        v = V[:tmax].copy(); fire = np.zeros((tmax, H), bool)
        for k in range(1, K + 1):
            exact = np.zeros((tmax, H)); bounded = np.zeros((tmax, H))
            for wi, d in enumerate(D):
                if d >= k:
                    j = max(D) + k - d
                    exact += Spad[j:j + tmax] @ W[wi].T
                else:
                    bounded += Sset.astype(float) @ Wp[wi].T
            v = beta * v + net["I"][net["_b"], k:tmax + k] + exact + bounded
            thr = theta + ((rho ** k) * A[:tmax] + gamma * (rho ** (k - 1)) * S[:tmax] if adapt else 0.0)
            fire |= v >= thr
        new = fire & Sset
        if np.array_equal(new, Sset):
            break
        Sset = new
    actual = np.zeros((tmax, H), bool)
    for k in range(1, K + 1):
        actual |= S[k:tmax + k] > 0
    viol = int((~Sset & actual).sum())
    return (~Sset).reshape(tmax, P, C).all(-1), viol


def main():
    d, m = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    adapt = len(sys.argv) > 4 and sys.argv[4] == "adapt"
    net = load(d, m)
    import os
    if os.environ.get("CORES"):                                               # repartition (weights unchanged)
        net["P"] = int(os.environ["CORES"]); net["C"] = net["H"] // net["P"]
    T, H, P, C, N = net["T"], net["H"], net["P"], net["C"], net["N"]
    side = int(round(np.sqrt(P))); mean_hops = 2 * (side * side - 1) / (3 * side) if side > 1 else 0.0
    tot = dict(nu_base=0.0, nu_cert=0.0, syn=0.0, route=0.0, skipped=0, upd=0, windows=0, viol=0)
    Wnz = sum((np.abs(w) > 0) for w in net["W"]).astype(bool)             # any-tap connectivity, (post, pre)
    fan_cores = np.array([len(set(np.nonzero(Wnz[:, j])[0] // C)) for j in range(H)])
    fan_syn = np.array([sum(int((np.abs(w[:, j]) > 0).sum()) for w in net["W"]) for j in range(H)])
    for b in range(N):
        net["_b"] = b
        V, S, A = simulate(net, b)
        cert, viol = certified_cores(net, V, S, A, K, adapt); tot["viol"] += viol
        # which neurons receive input or spike at each step (active update)
        recv = np.zeros((T, H), bool)
        for w, dd in zip(net["W"], net["D"]):
            recv[dd:] |= (S[:-dd] @ (np.abs(w.T) > 0)) > 0
        active = recv | (S > 0) | (np.abs(net["I"][b]) > 1e-12)
        e_step = np.where(active, E_NU_ACT, E_NU_INACT)                    # (T, H)
        tot["nu_base"] += e_step[1:].sum()
        tot["syn"] += (S.sum(0) * fan_syn).sum() * E_SYN
        tot["route"] += (S.sum(0) * fan_cores).sum() * (E_SPK_TILE + mean_hops * E_HOP)
        for p in range(P):
            t = 1; cols = slice(p * C, (p + 1) * C)
            while t < T:
                tot["nu_cert"] += e_step[t, cols].sum(); tot["upd"] += C       # ordinary update at step t
                if t < cert.shape[0] and cert[t, p] and t + K < T:
                    # certified silent at t+1..t+K: skip those K updates, pay one catch-up at t+K
                    tot["nu_cert"] += C * E_NU_ACT
                    tot["skipped"] += K * C; tot["windows"] += 1
                    t += K + 1
                else:
                    t += 1
    base = tot["nu_base"] + tot["syn"] + tot["route"]; cert_e = tot["nu_cert"] + tot["syn"] + tot["route"]
    print(f"RESULT model={m} K={K} adapt={int(adapt)} P={P} samples={N} violations={tot['viol']} "
          f"neuron_updates_skipped={tot['skipped']/(tot['skipped']+tot['upd']):.3f} "
          f"neuron_energy_saved={1-tot['nu_cert']/tot['nu_base']:.3f} total_energy_saved={1-cert_e/base:.3f} "
          f"share_neuron={tot['nu_base']/base:.3f} share_syn={tot['syn']/base:.3f} share_route={tot['route']/base:.3f} "
          f"uJ_per_sample_base={base/N*1e6:.3f} uJ_per_sample_cert={cert_e/N*1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
