"""PILOT-003 (pre-registered in research/BRAINSTORM_003_NEUROMORPHIC.md):
how often are LIF neurons PROVABLY silent for K steps in a trained recurrent SNN?
Premise of N2*: conservative synchronization lookahead from neuron dynamics.
"""
import gzip, json, os, shutil, subprocess, time
import numpy as np
import torch
import h5py

SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED); np.random.seed(SEED)
DATA = os.path.expanduser("~/research/data/shd")
T, NIN, H, NOUT = 100, 700, 256, 20
BETA = float(np.exp(-10 / 20)); THETA = 1.0
KS = (2, 4, 8)
LOCAL = os.environ.get("LOCAL") == "1"   # PILOT-003b: ring-local recurrent connectivity
NP = int(os.environ.get("NP", "8"))          # partitions (cores); 8 x 32 by default
RATE_TARGET = float(os.environ.get("RATE_TARGET", "0.05"))   # firing-rate penalty threshold
CERT = float(os.environ.get("CERT_LAMBDA", "0"))   # PILOT-004: certified-silence loss weight
CERT_K = 4
ALT = os.environ.get("ALT", "none")              # ALT-001: none | clamp | l1 | rate


def local_mask():
    p = np.arange(H) // (H // NP)
    d = np.abs(p[:, None] - p[None, :])
    return torch.tensor(((d <= 1) | (d == NP - 1)).astype(np.float32))   # self + ring neighbours


def fetch(split):
    h5 = f"{DATA}/shd_{split}.h5"
    if not os.path.exists(h5):
        os.makedirs(DATA, exist_ok=True)
        gz = h5 + ".gz"
        subprocess.run(["wget", "-q", "-c", "--timeout=60", "--tries=5", "-O", gz,
                        f"https://zenkelab.org/datasets/shd_{split}.h5.gz"], check=True)
        with gzip.open(gz) as fi, open(h5, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(gz)
    with h5py.File(h5) as f:
        times, units = f["spikes"]["times"][:], f["spikes"]["units"][:]
        labels = f["labels"][:].astype(np.int64)
    X = np.zeros((len(labels), T, NIN), np.uint8)
    for n, (tt, uu) in enumerate(zip(times, units)):
        b = np.minimum((tt / 1.4 * T).astype(int), T - 1)   # SHD samples last up to ~1.4 s
        X[n, b, uu] = 1
    return X, labels


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v); return (v >= 0).float()
    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors; return g / (1 + 10 * v.abs()) ** 2


class RSNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.win = torch.nn.Linear(NIN, H, bias=False)
        self.wrec = torch.nn.Linear(H, H, bias=False)
        self.wout = torch.nn.Linear(H, NOUT, bias=False)
        torch.nn.init.normal_(self.win.weight, 0, 0.05); torch.nn.init.normal_(self.wrec.weight, 0, 0.03)
        self.register_buffer("mask", local_mask() if LOCAL else torch.ones(H, H))

    def forward(self, x, record=False):
        B = x.shape[0]
        v = torch.zeros(B, H); s = torch.zeros(B, H); u = torch.zeros(B, NOUT)
        iext = self.win(x)                                   # [B,T,H], known in advance
        out, V, S, nspk = 0, [], [], 0
        for t in range(T):
            v = BETA * v + iext[:, t] + torch.nn.functional.linear(s, self.wrec.weight * self.mask)
            s = Spike.apply(v - THETA)
            v = v - s * THETA                                # reset by subtraction
            u = BETA * u + self.wout(s); out = out + u
            nspk = nspk + s.mean()
            V.append(v); S.append(s)
        return out, iext, torch.stack(V, 1), torch.stack(S, 1), nspk / T


def train(Xtr, ytr, Xte, yte, epochs=15):
    m = RSNN(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        perm = np.random.permutation(len(ytr)); t0 = time.time(); rates = []
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            x = torch.tensor(Xtr[idx], dtype=torch.float32); y = torch.tensor(ytr[idx])
            out, iext, V, S, rate = m(x)
            # mild, standard firing-rate regularization: penalize only rates above 5%
            if ALT == "rate":                                    # ALT-001: strong rate penalty instead
                loss = torch.nn.functional.cross_entropy(out, y) + 10.0 * torch.relu(rate - 0.01)
            else:
                loss = torch.nn.functional.cross_entropy(out, y) + torch.relu(rate - RATE_TARGET)
            if CERT > 0:
                loss = loss + CERT * cert_penalty(m, V, S, iext)
            if ALT == "l1":                                      # ALT-001: L1 on excitatory drive
                loss = loss + 0.1 * (torch.relu(m.wrec.weight) * m.mask).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if ALT == "clamp":                                   # ALT-001: hard budget R_i <= 0.33
                with torch.no_grad():
                    W = m.wrec.weight; R = (torch.relu(W) * m.mask).sum(1)
                    scale = torch.clamp(0.33 / R.clamp_min(1e-12), max=1.0)
                    W.copy_(torch.where(W > 0, W * scale[:, None], W))
        acc = evaluate(m, Xte, yte)
        print(f"epoch {ep} test acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return m, acc


def cert_penalty(m, V, S, iext, K=CERT_K):
    """Certified-silence loss (PILOT-004): worst-case K-step membrane reach, using the naive
    differentiable bound R_i = sum_j relu(W_rec[i,j]) for every step, penalized above threshold
    only where the neuron is actually silent in (t, t+K] (silence mask is detached)."""
    R = (torch.relu(m.wrec.weight) * m.mask).sum(1)          # [H]
    tmax = V.shape[1] - K
    silent = torch.ones_like(V[:, :tmax]).detach()
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k].detach())
        reach = BETA * reach + iext[:, k:tmax + k] + R
        worst = torch.maximum(worst, reach)
    return (torch.relu(worst - THETA) * silent).mean()


@torch.no_grad()
def evaluate(m, X, y):
    c = 0
    for i in range(0, len(y), 256):
        out = m(torch.tensor(X[i:i + 256], dtype=torch.float32))[0]
        c += (out.argmax(1).numpy() == y[i:i + 256]).sum()
    return c / len(y)


@torch.no_grad()
def horizons(m, X):
    W = (m.wrec.weight * m.mask).numpy()                     # W[i,k]: k -> i (masked if LOCAL)
    Wp = np.maximum(W, 0)
    _, iext, V, S, _ = m(torch.tensor(X, dtype=torch.float32))
    iext, V, S = iext.numpy(), V.numpy(), S.numpy()          # [B,T,H]
    B = V.shape[0]
    res = {}
    for K in KS:
        tmax = T - K
        rec1 = np.einsum("btk,ik->bti", S[:, :tmax], W)       # exact recurrent input at t+1
        def may_fire(R):                                     # R: [B,tmax,H] bound for m>=2
            v = V[:, :tmax].copy(); fire = np.zeros_like(v, bool)
            for k in range(1, K + 1):
                inp = iext[:, k:tmax + k] + (rec1 if k == 1 else R)
                v = BETA * v + inp
                fire |= v >= THETA
            return fire
        R_naive = np.broadcast_to(Wp.sum(1), (B, tmax, H))
        f_naive = may_fire(R_naive)
        Sset = np.ones((B, tmax, H), bool)                   # recursive: greatest fixed point
        for _ in range(50):
            R = np.einsum("btk,ik->bti", Sset.astype(np.float32), Wp)
            new = may_fire(R) & Sset
            if np.array_equal(new, Sset): break
            Sset = new
        actual = np.zeros((B, tmax, H), bool)                # oracle: really spikes in (t, t+K]
        for k in range(1, K + 1):
            actual |= S[:, k:tmax + k] > 0
        def stats(fire):
            silent = ~fire
            part = silent.reshape(B, tmax, NP, H // NP).all(-1)
            nbr = np.roll(part, 1, -1) & np.roll(part, -1, -1)       # both ring neighbours silent
            return dict(neuron=float(silent.mean()), partition32=float(part.mean()),
                        neighbours_silent=float(nbr.mean()),
                        layer=float(silent.all(-1).mean()))
        res[K] = dict(naive=stats(f_naive), recursive=stats(Sset), oracle=stats(actual),
                      fixed_point_iters=_ + 1)
    rate = float(S.mean())
    return res, rate


@torch.no_grad()
def integrity_checks(m, Xte, yte, Xs, K=4):
    """Interrogate PILOT-004 success: (1) soundness: certified-silent neuron-windows that
    actually spiked must be 0; (2) is recurrence still used? accuracy with recurrent weights
    zeroed / with only positive recurrent weights zeroed; (3) recurrent drive vs threshold."""
    W = (m.wrec.weight * m.mask).numpy(); Wp = np.maximum(W, 0)
    _, iext, V, S, _ = m(torch.tensor(Xs, dtype=torch.float32))
    iext, V, S = iext.numpy(), V.numpy(), S.numpy(); B = V.shape[0]; tmax = T - K
    rec1 = np.einsum("btk,ik->bti", S[:, :tmax], W)
    def may_fire(R):
        v = V[:, :tmax].copy(); fire = np.zeros_like(v, bool)
        for k in range(1, K + 1):
            v = BETA * v + iext[:, k:tmax + k] + (rec1 if k == 1 else R); fire |= v >= THETA
        return fire
    Sset = np.ones((B, tmax, H), bool)
    for _ in range(50):
        new = may_fire(np.einsum("btk,ik->bti", Sset.astype(np.float32), Wp)) & Sset
        if np.array_equal(new, Sset): break
        Sset = new
    actual = np.zeros((B, tmax, H), bool)
    for k in range(1, K + 1): actual |= S[:, k:tmax + k] > 0
    viol = int((~Sset & actual).sum())
    full = evaluate(m, Xte, yte)
    w0 = m.wrec.weight.clone()
    m.wrec.weight.zero_(); acc_norec = evaluate(m, Xte, yte)
    m.wrec.weight.copy_(torch.clamp(w0, max=0)); acc_noexc = evaluate(m, Xte, yte)
    m.wrec.weight.copy_(w0)
    R = Wp.sum(1)
    print(f"INTEGRITY K={K}: certified-silent-but-spiked windows = {viol} (must be 0)")
    print(f"INTEGRITY accuracy: full {full:.3f} | recurrent zeroed {acc_norec:.3f} | excitatory recurrent zeroed {acc_noexc:.3f}")
    print(f"INTEGRITY R_i = sum relu(W_rec): mean {R.mean():.4f} median {np.median(R):.4f} max {R.max():.4f} (theta={THETA}); |W_rec| mean {np.abs(W).sum(1).mean():.4f}")


def main():
    Xtr, ytr = fetch("train"); Xte, yte = fetch("test")
    print("data", Xtr.shape, Xte.shape, flush=True)
    m, acc = train(Xtr, ytr, Xte, yte)
    sub = np.random.default_rng(1).choice(len(yte), 500, replace=False)
    res, rate = horizons(m, Xte[sub])
    if os.environ.get("SAVE"):
        torch.save(m.state_dict(), os.path.expanduser(os.environ["SAVE"]))
        integrity_checks(m, Xte, yte, Xte[sub])
    print(f"\ntest accuracy {acc:.3f} | mean firing rate {rate:.4f} spikes/neuron/step")
    for K, r in res.items():
        print(f"K={K}: " + " | ".join(f"{b}: neuron {r[b]['neuron']:.3f} part32 {r[b]['partition32']:.3f} "
                                       f"layer {r[b]['layer']:.3f} nbr {r[b]['neighbours_silent']:.3f}" for b in ("naive", "recursive", "oracle"))
              + f" | fp iters {r['fixed_point_iters']}")
    json.dump(dict(accuracy=float(acc), rate=rate, results=res), open(os.environ["OUTJSON"], "w") if os.environ.get("OUTJSON") else open((f"../results/pilot004_l{CERT:g}_s{SEED}.json" if SEED or CERT > 0 else "../results/pilot003b.json") if LOCAL else "../results/pilot003.json", "w"), indent=1)


if __name__ == "__main__":
    main()
