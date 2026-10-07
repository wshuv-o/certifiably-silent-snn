"""DCLS-001: per-synapse LEARNABLE delays, with the silence certificate on top.

WHY THIS AND NOT MORE TAPS.
DELAY-002 established that multi-tap delays raise accuracy (+6.42 points) but that stacking taps onto
width collapses: 4 taps at H=1024 is 4.19M recurrent parameters against ~7k training samples and scored
73.74%, while every ~1M-parameter configuration scored 77-86%. The model is **data-limited at ~1M
recurrent parameters**, so capacity cannot buy accuracy. Multi-tap multiplies weights by the tap count;
the published ~95% SHD methods instead learn **one delay per synapse**, keeping the weight count at H^2.
That is the difference this file implements.

PARAMETERISATION.
Each synapse (i,j) has a weight W_ij and a continuous delay D_ij in [DMIN, DMAX], via
    D = DMIN + (DMAX - DMIN) * sigmoid(Draw)
so delays stay in range and stay differentiable. The contribution of synapse (i,j) at step t is
W_ij * s_j(t - D_ij), with D_ij interpolated linearly onto the integer grid by a triangular kernel:
    Wk_ij = W_ij * relu(1 - |D_ij - k|)            for integer taps k = DMIN..DMAX
    drive_i(t) = sum_k sum_j Wk_ij s_j(t-k)
Each synapse therefore lands on at most two adjacent integer taps and the kernel sums to 1, so total
synaptic strength is preserved. **Parameter count is 2*H^2 (W and Draw) regardless of DMAX** -- fewer
than the 3-tap model's 3*H^2, with a far richer delay space. Compute is DMAX-DMIN+1 matmuls per step.

DMIN DEFAULTS TO 2, DELIBERATELY.
DELAY-002/003 found that omitting the unit-delay tap improves accuracy *and* certifiability at once, and
d_min = 2 gives two steps of exact free lookahead (horizons k <= 2 need no bound at all). Clamping
learnable delays to >= 2 preserves that guarantee by construction rather than hoping training finds it.

THE CERTIFICATE IS UNCHANGED IN FORM.
At horizon k, tap d arrives at ts = t+1+k-d; for d >= k that time is <= t and is used exactly, so only
taps with d < K bind a K-step certificate. With DMIN = 2 and K = 4 that is just taps 2 and 3, so
R_short = R_2 + R_3. The network can make R_short arbitrarily small by *learning* to push synapses to
d >= 4 -- the continuous version of the reallocation mechanism that DELAY-003 observed discretely
(R_per_delay [4.97,4.90,5.40] -> [0.28,6.03,6.46] on SHD, [5.73,6.12,7.19] -> [0.29,8.56,9.98] on SSC).

Env: DMIN, DMAX, H, SEED, EPOCHS, CERT_LAMBDA, TAUM, AUG, INIT, TEST, LR, RAMP, DATASET, SAVE.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_silence import fetch, T, NIN, NOUT

DATASET = os.environ.get("DATASET", "shd")
if DATASET == "ssc":
    NOUT = 35

dev = "cuda"
H = int(os.environ.get("H", "512")); CPC = 32; NPC = H // CPC
TAUM = float(os.environ.get("TAUM", "2.0"))
BETA = float(np.exp(-1.0 / TAUM)); THETA = 1.0
BETA_OUT = float(np.exp(-0.5))
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02
DMIN = int(os.environ.get("DMIN", "2"))
DMAX = int(os.environ.get("DMAX", "8"))
assert 1 <= DMIN <= DMAX
DEL = list(range(DMIN, DMAX + 1))
LAM = float(os.environ.get("CERT_LAMBDA", "0")); SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "150")); K = 4
AUG = int(os.environ.get("AUG", "2"))
TEST = os.environ.get("TEST", "0") == "1"; TAG = os.environ.get("TAG", "cfg")
VALSPK = [int(s) for s in os.environ.get("VALSPK", "3,6").split(",")]
INIT = os.environ.get("INIT", "")
LR = float(os.environ.get("LR", "2e-3")); RAMP = os.environ.get("RAMP", "0") == "1"
torch.manual_seed(SEED); np.random.seed(SEED)


class LazySpikes:
    def __init__(self, path, window=1.0):
        import h5py
        self.f = h5py.File(path, "r", locking=False); self.window = window
        self.times, self.units = self.f["spikes"]["times"], self.f["spikes"]["units"]
        self.labels = self.f["labels"][:].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def get(self, idx):
        idx = np.asarray(list(idx)); X = np.zeros((len(idx), T, NIN), np.uint8)
        for n, i in enumerate(idx):
            tt, uu = self.times[i], self.units[i]
            X[n, np.minimum((tt / self.window * T).astype(int), T - 1), uu] = 1
        return X


def fetch_ssc(split):
    h5 = os.path.expanduser(f"~/research/data/ssc/ssc_{split}.h5")
    assert os.path.exists(h5), f"missing {h5}"
    D = LazySpikes(h5); return D, D.labels


def batch(D, idx):
    return D[idx] if isinstance(D, np.ndarray) else D.get(idx)


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors; return g / (1 + 10 * x.abs()) ** 2


class DCLSNet(torch.nn.Module):
    """One recurrent layer with a learnable delay per synapse, interpolated onto integer taps."""

    def __init__(self):
        super().__init__()
        self.win = torch.nn.Linear(NIN, H, bias=False)
        torch.nn.init.normal_(self.win.weight, 0, 0.05)
        self.w = torch.nn.Parameter(torch.empty(H, H))
        torch.nn.init.normal_(self.w, 0, 0.02)
        # delays start spread uniformly over the allowed range (sigmoid(Draw) ~ U(0,1) in logit space)
        self.draw = torch.nn.Parameter(torch.empty(H, H).uniform_(-2.0, 2.0))
        self.wout = torch.nn.Linear(H, NOUT, bias=False)
        torch.nn.init.normal_(self.wout.weight, 0, 0.05)
        self.register_buffer("quiet", torch.ones(H))

    def delays(self):
        return DMIN + (DMAX - DMIN) * torch.sigmoid(self.draw)

    def taps(self):
        """Effective weight matrix per integer tap. Triangular interpolation; kernel sums to 1."""
        D = self.delays()
        return [self.w * torch.relu(1.0 - (D - float(k)).abs()) for k in DEL]

    def forward(self, x):
        Bsz = x.shape[0]
        Wt = self.taps()
        v = torch.zeros(Bsz, H, device=x.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(Bsz, NOUT, device=x.device)
        iext = self.win(x); out = 0; V, S = [], []
        hist = [torch.zeros_like(v) for _ in range(DMAX)]          # hist[j] = s(t-1-j)
        for t in range(T):
            a = RHO * a + GAMMA * s
            rec = 0
            for wi, d in enumerate(DEL):
                rec = rec + torch.nn.functional.linear(hist[d - 1], Wt[wi])
            v = BETA * v + iext[:, t] + rec
            thr = THETA + a
            s = Spike.apply(v - thr)
            v = v - s * thr
            u = BETA_OUT * u + self.wout(s); out = out + u
            V.append(v); S.append(s)
            hist = [s] + hist[:-1]
        return out, iext, torch.stack(V, 1), torch.stack(S, 1)


def delay_terms(S, Wt, Wp, Sset_f, k, tmax, Bsz):
    """Split the taps at horizon k into already-determined (d >= k, exact) and future (d < k, bounded)."""
    Spad = torch.cat([torch.zeros(Bsz, DMAX, H, device=S.device, dtype=S.dtype), S], 1)
    exact = 0; bounded = 0
    for wi, d in enumerate(DEL):
        if d >= k:
            j = DMAX + k - d
            exact = exact + Spad[:, j:j + tmax] @ Wt[wi].T
        else:
            bounded = bounded + Sset_f @ Wp[wi].T
    return exact, bounded


def cert_penalty(m, V, S, iext):
    Wt = m.taps(); Wp = [torch.relu(w) for w in Wt]
    Bsz = V.shape[0]; tmax = T - K
    ones = torch.ones(Bsz, tmax, H, device=V.device)
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9)
    silent = torch.ones_like(reach)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k].detach())
        exact, bounded = delay_terms(S.detach(), Wt, Wp, ones, k, tmax, Bsz)
        reach = BETA * reach + iext[:, k:tmax + k] + exact + bounded
        worst = torch.maximum(worst, reach)
    return (torch.relu(worst - THETA) * silent * m.quiet).mean()


def augment(x):
    if AUG >= 2:
        x = torch.roll(x, int(np.random.randint(-10, 11)), dims=1)
        x = torch.roll(x, int(np.random.randint(-10, 11)), dims=2)
        x = x.clone()
        c0 = int(np.random.randint(0, x.shape[2] - 40)); x[:, :, c0:c0 + int(np.random.randint(0, 41))] = 0
        t0 = int(np.random.randint(0, x.shape[1] - 10)); x[:, t0:t0 + int(np.random.randint(0, 11)), :] = 0
        return x
    x = torch.roll(x, int(np.random.randint(-5, 6)), dims=1)
    return torch.roll(x, int(np.random.randint(-5, 6)), dims=2)


@torch.no_grad()
def evaluate(m, X, y):
    c = 0
    for i in range(0, len(y), 256):
        xb = torch.tensor(batch(X, range(i, min(i + 256, len(y)))), dtype=torch.float32, device=dev)
        c += (m(xb)[0].argmax(1).cpu().numpy() == y[i:i + 256]).sum()
    return c / len(y)


@torch.no_grad()
def certify(m, X):
    Wt = m.taps(); Wp = [torch.relu(w) for w in Wt]
    res = {"core_cert": 0.0, "core_oracle": 0.0, "violations": 0, "neuron_cert": 0.0, "allcore_cert": 0.0}
    n = 0
    for i in range(0, len(X), 100):
        _, iext, V, S = m(torch.tensor(batch(X, range(i, min(i + 100, len(X)))), dtype=torch.float32, device=dev))
        Bsz = V.shape[0]; tmax = T - K

        def may_fire(Sset_f):
            v = V[:, :tmax].clone(); fire = torch.zeros_like(v, dtype=torch.bool)
            for k in range(1, K + 1):
                exact, bounded = delay_terms(S, Wt, Wp, Sset_f, k, tmax, Bsz)
                v = BETA * v + iext[:, k:tmax + k] + exact + bounded
                fire |= v >= THETA
            return fire

        Sset = torch.ones(Bsz, tmax, H, dtype=torch.bool, device=dev)
        for _ in range(50):
            new = may_fire(Sset.float()) & Sset
            if torch.equal(new, Sset): break
            Sset = new
        actual = torch.zeros(Bsz, tmax, H, dtype=torch.bool, device=dev)
        for k in range(1, K + 1): actual |= S[:, k:tmax + k] > 0
        cs = (~Sset).reshape(Bsz, tmax, NPC, CPC).all(-1)
        co = (~actual).reshape(Bsz, tmax, NPC, CPC).all(-1)
        res["core_cert"] += cs.float().mean().item() * Bsz
        res["core_oracle"] += co.float().mean().item() * Bsz
        res["allcore_cert"] += cs.float().mean().item() * Bsz
        res["neuron_cert"] += (~Sset).float().mean().item() * Bsz
        res["violations"] += int((~Sset & actual).sum().item()); n += Bsz
    for k in ("core_cert", "core_oracle", "neuron_cert", "allcore_cert"): res[k] /= n
    Rd = [float(torch.relu(w).sum(1).mean()) for w in Wt]
    D = m.delays()
    res["R_per_delay"] = Rd; res["delays"] = DEL
    res["R_mean"] = float(sum(Rd))
    res["R_short"] = float(sum(r for d, r in zip(DEL, Rd) if d < K))
    res["budget"] = (1 - BETA) * THETA
    # where did the learned delays end up? mass-weighted, since tiny weights are irrelevant
    wmass = torch.relu(m.w).detach()
    res["delay_mean"] = float(D.mean()); res["delay_mean_wtd"] = float((D * wmass).sum() / wmass.sum())
    res["frac_delay_lt_K"] = float((D < K).float().mean())
    return res


def main():
    if DATASET == "ssc":
        Xtr, ytr = fetch_ssc("train"); Xte, yte = fetch_ssc("test")
        rng = np.random.default_rng(12345)
        mon = rng.choice(len(ytr), min(1500, len(ytr) // 20), replace=False)
        Xva, yva = Xtr, ytr[mon]; vsel = mon
        tidx = np.setdiff1d(np.arange(len(ytr)), mon)
    else:
        Xtr, ytr = fetch("train"); Xte, yte = fetch("test")
        import h5py
        with h5py.File(os.path.expanduser('~/research/data/shd/shd_train.h5')) as f5:
            spk = f5['extra']['speaker'][:]
        isval = np.isin(spk, VALSPK)
        vidx, tidx0 = np.nonzero(isval)[0], np.nonzero(~isval)[0]
        Xva, yva = Xtr[vidx], ytr[vidx]; vsel = None
        Xtr, ytr = Xtr[tidx0], ytr[tidx0]; tidx = np.arange(len(ytr))
    m = DCLSNet().to(dev)
    if INIT:
        sd = {k: v for k, v in torch.load(os.path.expanduser(INIT), map_location=dev).items() if k != 'quiet'}
        print("INIT load:", m.load_state_dict(sd, strict=False), flush=True)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * ((len(tidx) + 127) // 128))
    t0 = time.time()

    def valacc():
        if vsel is None: return evaluate(m, Xva, yva)
        with torch.no_grad():
            pr = [m(torch.tensor(batch(Xva, vsel[i:i + 256]), dtype=torch.float32, device=dev))[0]
                  .argmax(1).cpu().numpy() for i in range(0, len(vsel), 256)]
        return float((np.concatenate(pr) == yva).mean())

    for ep in range(EPOCHS):
        perm = tidx[np.random.permutation(len(tidx))]
        for b in range(0, len(perm), 128):
            idx = perm[b:b + 128]
            x = augment(torch.tensor(batch(Xtr, idx), dtype=torch.float32, device=dev))
            yb = torch.tensor(ytr[idx], device=dev)
            out, iext, V, S = m(x)
            loss = torch.nn.functional.cross_entropy(out, yb)
            if LAM > 0:
                lam_eff = LAM * (min(1.0, (ep + 1) / max(1, EPOCHS // 2)) if RAMP else 1.0)
                loss = loss + lam_eff * cert_penalty(m, V, S, iext)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            print(f"epoch {ep} val acc {valacc():.4f} ({(time.time()-t0)/(ep+1):.0f}s/epoch)", flush=True)
    acc_val = valacc()
    if vsel is None:
        vsub = np.random.default_rng(0).choice(len(yva), min(300, len(yva)), replace=False)
        cert = certify(m, Xva[vsub])
    else:
        cert = certify(m, batch(Xva, vsel[:300]))
    acc = evaluate(m, Xte, yte) if TEST else float('nan')
    cert_test = certify(m, batch(Xte, np.random.default_rng(1).choice(len(yte), 300, replace=False))) if TEST else {}
    res = dict(tag=TAG, dataset=DATASET, seed=SEED, epochs=EPOCHS, H=H, taum=TAUM, aug=AUG, lam=LAM,
               dmin=DMIN, dmax=DMAX, acc_val=float(acc_val), acc=float(acc),
               test=({'acc': float(acc), **{('test_' + k): v for k, v in cert_test.items()}} if TEST else None),
               **cert)
    print("RESULT", json.dumps(res), flush=True)
    json.dump(res, open(f"../results/dc_{TAG}_s{SEED}.json", "w"), indent=1)
    if os.environ.get("SAVE"):
        torch.save(m.state_dict(), os.path.expanduser(os.environ["SAVE"]))


if __name__ == "__main__":
    main()
