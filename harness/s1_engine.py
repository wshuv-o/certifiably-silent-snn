"""S1 (pre-registered in research/N3_SCALEUP_PLAN.md): exact multi-process "neuromorphic core"
execution of a ring-local RSNN, comparing global barrier, local handshake, and certificate-based
synchronization, with emulated interconnect latency. Spikes must be bit-identical to a reference.
"""
import os, sys, time, json
os.environ["LOCAL"] = "1"
import numpy as np
import multiprocessing as mp
import torch
import pilot_silence as ps

P, C = 8, 32                      # cores, neurons per core
T = ps.T
KMAX = 16
BETA, THETA = ps.BETA, ps.THETA

# ---------------- model / data (module globals, inherited by fork) ----------------
G = {}


def load(path, X):
    m = ps.RSNN(); m.load_state_dict(torch.load(os.path.expanduser(path))); m.eval()
    W = (m.wrec.weight * m.mask).detach().numpy().astype(np.float64)        # [H,H], i <- j
    Win = m.win.weight.detach().numpy().astype(np.float64)                 # [H,NIN]
    I = np.einsum("btn,hn->bth", X.astype(np.float64), Win)               # [B,T,H] external input
    nbr = [((p - 1) % P, p, (p + 1) % P) for p in range(P)]
    Wloc = [np.concatenate([W[p*C:(p+1)*C, q*C:(q+1)*C] for q in nbr[p]], 1) for p in range(P)]  # [C,3C]
    Rloc = [np.maximum(w, 0).sum(1) for w in Wloc]                          # worst-case drive
    return dict(I=I, Wloc=Wloc, Rloc=Rloc, nbr=nbr)


def core_step(p, v, s_in, Iext):
    v = BETA * v + Iext + G["Wloc"][p] @ s_in
    s = (v >= THETA).astype(np.float64)
    return v - s * THETA, s


def horizon(p, v, b, t):
    """Lemma 1 certificate after step t: largest c<=KMAX with all neurons of core p provably
    silent in (t, t+c]; worst-case recurrent drive R for every future step (neighbours' next
    spikes unknown)."""
    I, R = G["I"][b], G["Rloc"][p]
    u = v.copy()
    for k in range(1, KMAX + 1):
        if t + k > T - 1:
            return k - 1
        u = BETA * u + I[t + k, p*C:(p+1)*C] + R
        if (u >= THETA).any():
            return k - 1
    return KMAX


def reference(b):
    v = [np.zeros(C) for _ in range(P)]; s = np.zeros((T + 1, P, C))
    for t in range(T - 1):
        for p in range(P):
            s_in = np.concatenate([s[t, q] for q in G["nbr"][p]])
            v[p], s[t + 1, p] = core_step(p, v[p], s_in, G["I"][b, t + 1, p*C:(p+1)*C])
    return s


# ---------------- worker ----------------
def worker(p, mode, L_ns, samples, SPK, PUB, CERT, bar, out_times, go):
    spk = np.frombuffer(SPK, dtype=np.float64).reshape(len(samples), T + 1, P, C)
    pub = np.frombuffer(PUB, dtype=np.int64).reshape(len(samples), P, T + 1)     # publish time (ns), 0 = not yet
    cert = np.frombuffer(CERT, dtype=np.int64).reshape(len(samples), P, T + 1)   # silent through step cert[.,q,k]
    left, _, right = G["nbr"][p]
    for si, b in enumerate(samples):
        bar.wait()                                               # align sample start (not timed per step)
        t0 = time.perf_counter_ns()
        v = np.zeros(C); last_c = 0
        seen = {left: 0, right: 0}                               # highest neighbour step known visible
        for t in range(T - 1):
            if mode == "barrier":
                if t > 0:
                    bar.wait()
            else:
                for q in (left, right):
                    while True:                                  # need neighbour's step-t spikes (or silence)
                        k = seen[q]
                        now = time.perf_counter_ns()
                        while k < T and pub[si, q, k + 1] and now >= pub[si, q, k + 1] + L_ns:
                            k += 1
                        seen[q] = k
                        if k >= t:
                            break
                        if mode == "cert" and k > 0 and cert[si, q, k] >= t:
                            break
                        if mode == "cert" and k == 0 and pub[si, q, 0] and now >= pub[si, q, 0] + L_ns \
                                and cert[si, q, 0] >= t:
                            break
            s_in = np.concatenate([spk[si, t, q] for q in G["nbr"][p]])
            v, s = core_step(p, v, s_in, G["I"][b, t + 1, p*C:(p+1)*C])
            spk[si, t + 1, p] = s
            if mode == "cert":
                if s.any() or t + 1 >= last_c:
                    last_c = t + 1 + horizon(p, v, b, t + 1) if not s.any() else t + 1
                cert[si, p, t + 1] = last_c
            pub[si, p, t + 1] = time.perf_counter_ns()          # publish after data is written
        out_times[si * P + p] = time.perf_counter_ns() - t0       # per core; sample time = slowest core


def run(mode, L_us, samples):
    n = len(samples)
    SPK = mp.RawArray("d", n * (T + 1) * P * C)
    PUB = mp.RawArray("q", n * P * (T + 1))
    CERT = mp.RawArray("q", n * P * (T + 1))
    pub = np.frombuffer(PUB, dtype=np.int64).reshape(n, P, T + 1)
    cert = np.frombuffer(CERT, dtype=np.int64).reshape(n, P, T + 1)
    pub[:, :, 0] = 1                                             # step 0 (all-zero state) visible from start
    cert[:, :, 0] = 0
    times = mp.RawArray("q", n * P)
    bar = mp.Barrier(P)
    procs = [mp.Process(target=worker, args=(p, mode, int(L_us * 1000), samples, SPK, PUB, CERT, bar, times, None))
             for p in range(P)]
    for pr in procs: pr.start()
    for pr in procs: pr.join()
    spk = np.frombuffer(SPK, dtype=np.float64).reshape(n, T + 1, P, C)
    return np.array(times[:]).reshape(n, P).max(1) / 1e6, spk   # ms per sample (slowest core)


def main():
    mp.set_start_method("fork")
    Xte, yte = ps.fetch("test")
    samples = list(np.random.default_rng(1).choice(len(yte), 200, replace=False))
    X = Xte[samples]
    rows = []
    for tag, path in (("control", "~/research/models/p005_s1_l0.pt"),
                      ("certified", "~/research/models/p005_s1_l0.1.pt")):
        G.clear(); G.update(load(path, X))
        local_ids = list(range(len(samples)))
        refs = np.stack([reference(b) for b in local_ids])     # [n,T+1,P,C]
        for L in (0, 5, 20):     # NOTE: barrier mode ignores L (conservative: favours the baseline)
            for mode in ("barrier", "handshake", "cert"):
                ms, spk = run(mode, L, local_ids)
                exact = bool(np.array_equal(spk[:, :T], refs[:, :T]))
                rows.append(dict(model=tag, L_us=L, mode=mode, median_ms=float(np.median(ms)),
                                 mean_ms=float(ms.mean()), exact=exact))
                print(rows[-1], flush=True)
    json.dump(rows, open("../results/s1_engine.json", "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n", df.pivot_table(index=["L_us", "mode"], columns="model", values="median_ms").round(3).to_string())
    print("ALL EXACT:", bool(df.exact.all()))


if __name__ == "__main__":
    main()
