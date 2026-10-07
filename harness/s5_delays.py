"""DELAY-001: recurrent SNN with multi-tap synaptic delays, and the silence certificate generalised
to delayed synapses.

WHY DELAYS (two independent reasons):
 1. Accuracy. Learned/structured delays are the single largest known factor on SHD: plain recurrent
    ALIF nets land near 83-84%, delay-based architectures reach ~95-96%. Our base model's ~77%
    validation ceiling is a property of the architecture class, not of the certificates
    (RADIUS-001 showed certification is already nearly free at every connectivity radius).
 2. The certificate gets STRONGER, not weaker. With v(t) = BETA*v(t-1) + iext(t) + sum_d W_d s(t-d),
    the k-th step ahead of a certificate origin needs spikes at times t+k-d. For d >= k that time is
    <= t, i.e. ALREADY KNOWN and exactly computable; only d < k contributes uncertainty. So the
    worst-case drive that must be bounded at horizon k is sum_{d<k} R_d, not the full sum_d R_d.
    In particular a minimum delay of d_min gives d_min steps of EXACT, free lookahead -- which is
    precisely the lookahead conservative PDES takes from minimum synaptic delay. Certificates then
    extend beyond it. With DELAYS=[1] this reduces exactly to the previous scheme (step 1 exact via
    the measured recurrent input, steps >= 2 bounded by R), so the implementation is a strict
    generalisation and must reproduce s4_improve.py at DELAYS=1.

Env: DELAYS (comma list, default "1"), H, SEED, EPOCHS, CERT_LAMBDA, TAUM, LOCAL, LOCAL_R, AUG,
     INIT, TEST, LR, RAMP, DATASET, SAVE.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_silence import fetch, T, NIN, NOUT

DATASET = os.environ.get("DATASET", "shd")        # shd | ssc | nmnist
if DATASET == "ssc":
    NOUT = 35                                     # Spiking Speech Commands has 35 classes
elif DATASET == "nmnist":
    # event-camera vision: 34x34 pixels, two polarities, ten digits
    NIN, NOUT = 34 * 34 * 2, 10

dev = "cuda"
H = int(os.environ.get("H", "512")); CPC = int(os.environ.get("CPC", "32")); NPC = H // CPC
TAUM = float(os.environ.get("TAUM", "2.0"))
BETA = float(np.exp(-1.0 / TAUM)); THETA = 1.0
BETA_OUT = float(np.exp(-0.5))
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02
DELAYS = sorted({int(d) for d in os.environ.get("DELAYS", "1").split(",")})
assert min(DELAYS) >= 1, "delays must be >= 1 step"
DMAX = max(DELAYS)
LAM = float(os.environ.get("CERT_LAMBDA", "0")); SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "40")); K = int(os.environ.get("K", "4"))
LOCAL = os.environ.get("LOCAL", "0") == "1"; LOCAL_R = int(os.environ.get("LOCAL_R", "1"))
AUG = int(os.environ.get("AUG", "1"))
TEST = os.environ.get("TEST", "0") == "1"; TAG = os.environ.get("TAG", "cfg")
VALSPK = [int(s) for s in os.environ.get("VALSPK", "3,6").split(",")]
INIT = os.environ.get("INIT", "")
LR = float(os.environ.get("LR", "2e-3")); RAMP = os.environ.get("RAMP", "0") == "1"
torch.manual_seed(SEED); np.random.seed(SEED)


class LazySpikes:
    """Bin spikes per batch straight from the h5 (SSC dense would be ~5 GB)."""

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
    d = os.path.expanduser("~/research/data/ssc")
    h5 = f"{d}/ssc_{split}.h5"
    assert os.path.exists(h5), f"missing {h5}; download it first"
    D = LazySpikes(h5)
    return D, D.labels


def fetch_nmnist(split):
    d = os.path.expanduser("~/research/data/nmnist")
    h5 = f"{d}/nmnist_{split}.h5"
    assert os.path.exists(h5), f"missing {h5}; run harness/make_nmnist_h5.py first"
    D = LazySpikes(h5)
    return D, D.labels


def batch(D, idx):
    return D[idx] if isinstance(D, np.ndarray) else D.get(idx)


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors; return g / (1 + 10 * x.abs()) ** 2


class DelayNet(torch.nn.Module):
    """One recurrent layer whose recurrence is a sum of delayed taps: sum_d W_d s(t-d)."""

    def __init__(self):
        super().__init__()
        # init order matches s4_improve.py (win, wrec, wout) so DELAYS=1 is as close to it as possible.
        # Exact bitwise reproduction is NOT expected: nn.Linear's default init consumes RNG that a raw
        # Parameter does not, so the equivalence check below is statistical, not bit-for-bit.
        self.win = torch.nn.Linear(NIN, H, bias=False)
        torch.nn.init.normal_(self.win.weight, 0, 0.05)
        # one weight matrix per delay tap; variance split across taps so total initial drive matches a 1-tap net
        self.wrec = torch.nn.ParameterList()
        for _ in DELAYS:
            w = torch.empty(H, H); torch.nn.init.normal_(w, 0, 0.02 / np.sqrt(len(DELAYS)))
            self.wrec.append(torch.nn.Parameter(w))
        self.wout = torch.nn.Linear(H, NOUT, bias=False)
        torch.nn.init.normal_(self.wout.weight, 0, 0.05)
        M = torch.ones(H, H)
        if LOCAL:
            pc = np.arange(H) // CPC
            d = np.abs(pc[:, None] - pc[None, :])
            M = M * torch.tensor(((d <= LOCAL_R) | (d >= NPC - LOCAL_R)).astype(np.float32))
        self.register_buffer("mask", M)
        self.register_buffer("quiet", torch.ones(H))

    def masked(self):
        return [w * self.mask for w in self.wrec]

    def forward(self, x):
        Bsz = x.shape[0]
        v = torch.zeros(Bsz, H, device=x.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(Bsz, NOUT, device=x.device)
        iext = self.win(x); out = 0; V, S = [], []
        hist = [torch.zeros_like(v) for _ in range(DMAX)]     # hist[j] = s(t-1-j)
        Wm = self.masked()
        for t in range(T):
            a = RHO * a + GAMMA * s
            rec = 0
            for wi, d in enumerate(DELAYS):
                rec = rec + torch.nn.functional.linear(hist[d - 1], Wm[wi])
            v = BETA * v + iext[:, t] + rec
            thr = THETA + a
            s = Spike.apply(v - thr)
            v = v - s * thr
            u = BETA_OUT * u + self.wout(s); out = out + u
            V.append(v); S.append(s)
            hist = [s] + hist[:-1]
        return out, iext, torch.stack(V, 1), torch.stack(S, 1)


def delay_terms(S, Wm, Wp, Sset_f, k, tmax, Bsz):
    """Recurrent drive at horizon k of a certificate window.

    Splits the delayed taps into the part that is already determined (d >= k, arriving from times <= t,
    so computed exactly from measured spikes) and the part that is still uncertain (d < k, bounded by
    the may-fire set). Returns (exact_part, bounded_part).
    """
    Spad = torch.cat([torch.zeros(Bsz, DMAX, H, device=S.device, dtype=S.dtype), S], 1)
    exact = 0; bounded = 0
    for wi, d in enumerate(DELAYS):
        if d >= k:                                   # arrival time t+k-d <= t: known exactly
            j = DMAX + k - d                         # index into the padded spike history
            exact = exact + Spad[:, j:j + tmax] @ Wm[wi].T
        else:                                        # arrival time > t: bound it
            bounded = bounded + Sset_f @ Wp[wi].T
    return exact, bounded


def cert_penalty(m, V, S, iext):
    """Training penalty: push the worst-case K-step reach below threshold on genuinely silent windows."""
    Wm = m.masked(); Wp = [torch.relu(w) for w in Wm]
    Bsz = V.shape[0]; tmax = T - K
    ones = torch.ones(Bsz, tmax, H, device=V.device)
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9)
    silent = torch.ones_like(reach)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k].detach())
        exact, bounded = delay_terms(S.detach(), Wm, Wp, ones, k, tmax, Bsz)
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
    """Lemma-2 fixed-point certificate, generalised to delayed synapses."""
    Wm = m.masked(); Wp = [torch.relu(w) for w in Wm]
    res = {"core_cert": 0.0, "core_oracle": 0.0, "violations": 0, "neuron_cert": 0.0, "allcore_cert": 0.0}
    n = 0
    for i in range(0, len(X), 100):
        _, iext, V, S = m(torch.tensor(batch(X, range(i, min(i + 100, len(X)))), dtype=torch.float32, device=dev))
        Bsz = V.shape[0]; tmax = T - K

        def may_fire(Sset_f):
            v = V[:, :tmax].clone(); fire = torch.zeros_like(v, dtype=torch.bool)
            for k in range(1, K + 1):
                exact, bounded = delay_terms(S, Wm, Wp, Sset_f, k, tmax, Bsz)
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
    Rd = [float(torch.relu(w).sum(1).mean()) for w in Wm]
    res["R_per_delay"] = Rd
    res["R_mean"] = float(sum(Rd))                                    # unbounded-horizon worst case
    res["R_short"] = float(sum(r for d, r in zip(DELAYS, Rd) if d < K))  # what binds a K-step certificate
    res["budget"] = (1 - BETA) * THETA
    res["delays"] = DELAYS
    return res


def main():
    if DATASET in ("ssc", "nmnist"):
        # Out-of-sample architecture test: the recipe is FROZEN from SHD, so nothing is selected here.
        # Neither set has speaker metadata, so a small random slice of train is held out for PROGRESS
        # MONITORING ONLY -- it is never used to choose anything. The test set is touched once.
        _fetch = fetch_ssc if DATASET == "ssc" else fetch_nmnist
        Xtr, ytr = _fetch("train"); Xte, yte = _fetch("test")
        rng = np.random.default_rng(12345)
        mon = rng.choice(len(ytr), min(1500, len(ytr) // 20), replace=False)
        Xva, yva = Xtr, ytr[mon]          # lazy dataset: index via batch() at use sites
        vsel = mon
        tidx = np.setdiff1d(np.arange(len(ytr)), mon)
        assert int(ytr.max()) + 1 <= NOUT, f"label count {int(ytr.max())+1} exceeds NOUT={NOUT}"
    else:
        Xtr, ytr = fetch("train"); Xte, yte = fetch("test")
        import h5py
        with h5py.File(os.path.expanduser('~/research/data/shd/shd_train.h5')) as f5:
            spk = f5['extra']['speaker'][:]
        isval = np.isin(spk, VALSPK)
        vidx, tidx = np.nonzero(isval)[0], np.nonzero(~isval)[0]
        Xva, yva = Xtr[vidx], ytr[vidx]; vsel = None
        Xtr, ytr = Xtr[tidx], ytr[tidx]; tidx = np.arange(len(ytr))
    m = DelayNet().to(dev)
    if INIT:
        sd = {k: v for k, v in torch.load(os.path.expanduser(INIT), map_location=dev).items()
              if k not in ('mask', 'quiet')}
        print("INIT load:", m.load_state_dict(sd, strict=False), flush=True)
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
    steps = EPOCHS * ((len(ytr) + 127) // 128)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps))
    t0 = time.time()
    def valacc():
        if vsel is None: return evaluate(m, Xva, yva)
        return float((np.concatenate([m(torch.tensor(batch(Xva, vsel[i:i + 256]), dtype=torch.float32,
                    device=dev))[0].argmax(1).cpu().numpy() for i in range(0, len(vsel), 256)]) == yva).mean())

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
    res = dict(tag=TAG, dataset=DATASET, seed=SEED, epochs=EPOCHS, H=H, K=K, cpc=CPC, taum=TAUM, local=int(LOCAL),
               local_r=LOCAL_R, aug=AUG, lam=LAM, acc_val=float(acc_val), acc=float(acc),
               test=({'acc': float(acc), **{('test_' + k): v for k, v in cert_test.items()}} if TEST else None),
               **cert)
    print("RESULT", json.dumps(res), flush=True)
    name = f"d1_{TAG}_s{SEED}"
    json.dump(res, open(f"../results/{name}.json", "w"), indent=1)
    if os.environ.get("SAVE"):
        torch.save(m.state_dict(), os.path.expanduser(os.environ["SAVE"]))


if __name__ == "__main__":
    main()
