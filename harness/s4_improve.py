"""SWEEP-002 (pre-registered in research/N3_SCALEUP_PLAN.md): s3_sweep.py + speaker-disjoint
validation (VALSPK), fine-tuning from a checkpoint (INIT, LR, ramped lambda) and knowledge
distillation (TEACHER, KDW, KDT).

SWEEP-001 (pre-registered in research/N3_SCALEUP_PLAN.md). Derived from s2_strong.py.
Adds: validation split (VAL), sink hub cores (NHUB), activity-weighted recursive-aware loss (ALPHA),
L1 weight (MU). Selection uses validation only; TEST=1 only for confirmation runs.

Original: S2 (pre-registered in research/N3_SCALEUP_PLAN.md): certified-silence training on a stronger
SNN — one dense recurrent layer of 512 adaptive LIF (ALIF) neurons on SHD, GPU.
Certificates use the BASE threshold theta (adaptation only raises the threshold -> sound).
Env: CERT_LAMBDA (default 0), SEED (default 1), EPOCHS (default 40), SAVE (optional path).
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from pilot_silence import fetch, T, NIN, NOUT                     # same SHD binning (100 bins, 1.4 s)

dev = "cuda"
H = int(os.environ.get("H", "512")); CPC = 32; NPC = H // CPC    # neurons, cores, neurons per core (SCALE-001)
TAUM = float(os.environ.get("TAUM", "2.0"))          # membrane time constant in steps; 2.0 = original
BETA = float(np.exp(-1.0 / TAUM)); THETA = 1.0           # budget (1-BETA)*THETA grows as TAUM falls
BETA_OUT = float(np.exp(-0.5))                           # readout integrator: HELD FIXED so only the
                                                         # membrane (and hence the budget) is varied
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02                     # adaptation decay / jump
LOCAL = os.environ.get("LOCAL", "0") == "1"             # ring-local recurrent connectivity
LOCAL_R = int(os.environ.get("LOCAL_R", "1"))           # neighbourhood radius in cores
LAM = float(os.environ.get("CERT_LAMBDA", "0")); SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "40")); K = 4
DATASET = os.environ.get("DATASET", "shd")                     # shd | ssc (SSC-001)
ALT = os.environ.get("ALT", "none")                             # none | clamp | l1
NHUB = int(os.environ.get("NHUB", "0"))                         # sink hub cores (A)
ALPHA = float(os.environ.get("ALPHA", "1"))                     # 1 = standard loss; <1 recursive-aware (C)
MU = float(os.environ.get("MU", "0.1"))                         # L1 weight
TEST = os.environ.get("TEST", "0") == "1"; TAG = os.environ.get("TAG", "cfg")
VALSPK = [int(s) for s in os.environ.get("VALSPK", "3,6").split(",")]
INIT = os.environ.get("INIT", ""); TEACHER = os.environ.get("TEACHER", "")
KDW = float(os.environ.get("KDW", "1.0")); KDT = float(os.environ.get("KDT", "2.0"))
LR = float(os.environ.get("LR", "2e-3")); RAMP = os.environ.get("RAMP", "0") == "1"
if DATASET == "ssc":
    NOUT = 35
torch.manual_seed(SEED); np.random.seed(SEED)


class LazySpikes:
    """SSC-001: bin spikes per batch from the h5 file (the full dense array would need ~5 GB)."""
    def __init__(self, path, window=1.0):
        import h5py
        self.f = h5py.File(path, "r"); self.window = window
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
    import gzip, shutil, subprocess
    d = os.path.expanduser("~/research/data/ssc"); os.makedirs(d, exist_ok=True)
    h5 = f"{d}/ssc_{split}.h5"
    if not os.path.exists(h5):
        subprocess.run(["wget", "-q", "-c", "--timeout=60", "--tries=10", "-O", h5 + ".gz",
                        f"https://zenkelab.org/datasets/ssc_{split}.h5.gz"], check=True)
        with gzip.open(h5 + ".gz") as fi, open(h5, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(h5 + ".gz")
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


class ALIFNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.win = torch.nn.Linear(NIN, H, bias=False)
        self.wrec = torch.nn.Linear(H, H, bias=False)
        self.wout = torch.nn.Linear(H, NOUT, bias=False)
        torch.nn.init.normal_(self.win.weight, 0, 0.05); torch.nn.init.normal_(self.wrec.weight, 0, 0.02)
        torch.nn.init.normal_(self.wout.weight, 0, 0.05)
        hub = torch.zeros(H, dtype=torch.bool); hub[:NHUB * CPC] = True
        M = torch.ones(H, H); M[~hub][:, hub] = 0.0
        M[(~hub).nonzero().squeeze(1)[:, None], hub.nonzero().squeeze(1)[None, :]] = 0.0
        if LOCAL:                                        # restrict recurrence to neighbouring cores
            pc = np.arange(H) // CPC                     # core index of each neuron
            d = np.abs(pc[:, None] - pc[None, :])
            M = M * torch.tensor(((d <= LOCAL_R) | (d >= NPC - LOCAL_R)).astype(np.float32))
        self.register_buffer("mask", M); self.register_buffer("quiet", (~hub).float())

    def forward(self, x):
        Bsz = x.shape[0]
        v = torch.zeros(Bsz, H, device=x.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(Bsz, NOUT, device=x.device)
        iext = self.win(x); out = 0; V, S = [], []
        for t in range(T):
            a = RHO * a + GAMMA * s
            v = BETA * v + iext[:, t] + torch.nn.functional.linear(s, self.wrec.weight * self.mask)
            thr = THETA + a
            s = Spike.apply(v - thr)
            v = v - s * thr
            u = BETA_OUT * u + self.wout(s); out = out + u
            V.append(v); S.append(s)
        return out, iext, torch.stack(V, 1), torch.stack(S, 1)


def cert_penalty(m, V, S, iext):
    Wp = torch.relu(m.wrec.weight) * m.mask
    R = Wp.sum(1)
    tmax = T - K; silent = torch.ones_like(V[:, :tmax])
    if ALPHA < 1:
        act = S[:, 1:tmax + 1].detach()
        for k in range(2, K + 1): act = torch.maximum(act, S[:, k:tmax + k].detach())
        R = ALPHA * R + (1 - ALPHA) * (act @ Wp.T)
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k].detach())
        reach = BETA * reach + iext[:, k:tmax + k] + R
        worst = torch.maximum(worst, reach)
    return (torch.relu(worst - THETA) * silent * m.quiet).mean()


def augment(x):
    x = torch.roll(x, int(np.random.randint(-5, 6)), dims=1)      # time shift (circular, small)
    return torch.roll(x, int(np.random.randint(-5, 6)), dims=2)   # channel jitter


@torch.no_grad()
def evaluate(m, X, y):
    c = 0
    for i in range(0, len(y), 256):
        out = m(torch.tensor(batch(X, range(i, min(i + 256, len(y)))), dtype=torch.float32, device=dev))[0]
        c += (out.argmax(1).cpu().numpy() == y[i:i + 256]).sum()
    return c / len(y)


@torch.no_grad()
def certify(m, X):
    """Recursive certificate (Lemma 2) on GPU; dense recurrence; base threshold."""
    W = m.wrec.weight * m.mask; Wp = torch.relu(W)
    res = {"core_cert": 0.0, "core_oracle": 0.0, "violations": 0, "neuron_cert": 0.0}
    n = 0
    for i in range(0, len(X), 100):
        _, iext, V, S = m(torch.tensor(X[i:i + 100], dtype=torch.float32, device=dev))
        Bsz = V.shape[0]; tmax = T - K
        rec1 = S[:, :tmax] @ W.T
        def may_fire(R):
            v = V[:, :tmax].clone(); fire = torch.zeros_like(v, dtype=torch.bool)
            for k in range(1, K + 1):
                v = BETA * v + iext[:, k:tmax + k] + (rec1 if k == 1 else R); fire |= v >= THETA
            return fire
        Sset = torch.ones(Bsz, tmax, H, dtype=torch.bool, device=dev)
        for _ in range(50):
            new = may_fire(Sset.float() @ Wp.T) & Sset
            if torch.equal(new, Sset): break
            Sset = new
        actual = torch.zeros(Bsz, tmax, H, dtype=torch.bool, device=dev)
        for k in range(1, K + 1): actual |= S[:, k:tmax + k] > 0
        cs = (~Sset).reshape(Bsz, tmax, NPC, CPC).all(-1); co = (~actual).reshape(Bsz, tmax, NPC, CPC).all(-1)
        q0 = NHUB  # cores others wait on = non-hub cores
        res["core_cert"] += cs[..., q0:].float().mean().item() * Bsz; res["core_oracle"] += co[..., q0:].float().mean().item() * Bsz
        res["allcore_cert"] = res.get("allcore_cert", 0) + cs.float().mean().item() * Bsz
        res["neuron_cert"] += (~Sset).float().mean().item() * Bsz
        res["violations"] += int((~Sset & actual).sum().item()); n += Bsz
    for k in ("core_cert", "core_oracle", "neuron_cert", "allcore_cert"): res[k] /= n
    res["R_mean"] = float(Wp.sum(1).mean()); res["budget"] = (1 - BETA) * THETA
    return res


def main():
    if DATASET == "ssc":
        Xtr, ytr = fetch_ssc("train"); Xte, yte = fetch_ssc("test")
    else:
        Xtr, ytr = fetch("train"); Xte, yte = fetch("test")
    import h5py
    with h5py.File(os.path.expanduser('~/research/data/shd/shd_train.h5')) as f5:
        spk = f5['extra']['speaker'][:]
    isval = np.isin(spk, VALSPK); vidx, tidx = np.nonzero(isval)[0], np.nonzero(~isval)[0]   # speaker-disjoint
    Xva, yva = batch(Xtr, vidx), ytr[vidx]
    Xtr = Xtr[tidx] if isinstance(Xtr, np.ndarray) else Xtr; ytr_full = ytr; ytr = ytr[tidx]
    m = ALIFNet().to(dev)
    if os.environ.get("COMPILE") == "1":        # launch-bound model: fuse kernels + CUDA graphs.
        m = torch.compile(m, mode="reduce-overhead")   # validate numerics before trusting results
    if INIT:                                         # fine-tune: keep this model's own mask/quiet buffers
        sd = {k: v for k, v in torch.load(os.path.expanduser(INIT), map_location=dev).items() if k not in ('mask', 'quiet')}
        m.load_state_dict(sd, strict=False)
    teacher = None
    if TEACHER:
        teacher = ALIFNet().to(dev); tsd = torch.load(os.path.expanduser(TEACHER), map_location=dev)
        tsd['mask'] = torch.ones(H, H, device=dev); tsd['quiet'] = torch.ones(H, device=dev)
        teacher.load_state_dict(tsd); teacher.eval()
    opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
    total_steps = EPOCHS * ((len(ytr) + 127) // 128); step = 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * ((len(ytr) + 127) // 128))
    for ep in range(EPOCHS):
        perm = np.random.permutation(len(ytr)); t0 = time.time()
        m.train()
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            x = augment(torch.tensor(batch(Xtr, idx), dtype=torch.float32, device=dev))
            y = torch.tensor(ytr[idx], device=dev)
            out, iext, V, S = m(x)
            loss = torch.nn.functional.cross_entropy(out, y) + torch.relu(S.mean() - 0.05)
            lam_eff = LAM * (min(1.0, step / (0.5 * total_steps)) if RAMP else 1.0); step += 1
            if LAM > 0:
                loss = loss + lam_eff * cert_penalty(m, V, S, iext)
            if teacher is not None:
                with torch.no_grad():
                    tout = teacher(x)[0]
                loss = loss + KDW * KDT * KDT * torch.nn.functional.kl_div(
                    torch.log_softmax(out / KDT, 1), torch.softmax(tout / KDT, 1), reduction='batchmean')
            if ALT == "l1":                                      # ALT-001: L1 on excitatory drive
                loss = loss + MU * ((torch.relu(m.wrec.weight) * m.mask).sum(1) * m.quiet).sum() / m.quiet.sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ALT == "clamp":                                   # ALT-001: hard budget R_i <= 0.33
                with torch.no_grad():
                    W = m.wrec.weight; R = (torch.relu(W) * m.mask).sum(1)
                    scale = torch.where(m.quiet > 0, torch.clamp(0.33 / R.clamp_min(1e-12), max=1.0), torch.ones_like(R))
                    W.copy_(torch.where(W > 0, W * scale[:, None], W))
        if ep % 5 == 4 or ep == EPOCHS - 1:
            m.eval(); print(f"epoch {ep} val acc {evaluate(m, Xva, yva):.4f} ({time.time() - t0:.0f}s/epoch)", flush=True)
    m.eval()
    acc_val = evaluate(m, Xva, yva)
    vsub = np.random.default_rng(1).choice(len(yva), min(500, len(yva)), replace=False)
    cert = certify(m, Xva[vsub])
    acc = evaluate(m, Xte, yte) if TEST else float('nan')
    if TEST:
        sub = np.random.default_rng(1).choice(len(yte), 500, replace=False)
        cert_test = certify(m, batch(Xte, sub))
    else:
        cert_test = {}
    sub = vsub; Xte, yte = Xva, yva
    w0 = m.wrec.weight.data.clone(); m.wrec.weight.data.zero_(); acc_norec = evaluate(m, Xte, yte); m.wrec.weight.data.copy_(w0)
    _, _, _, S = m(torch.tensor(batch(Xte, sub[:100]), dtype=torch.float32, device=dev))
    res = dict(tag=TAG, dataset=DATASET, alt=ALT, lam=LAM, mu=MU, nhub=NHUB, alpha=ALPHA, seed=SEED, epochs=EPOCHS,
               acc_val=float(acc_val), test=({'acc': float(acc), **{('test_' + k): v for k, v in cert_test.items()}} if TEST else None), acc=float(acc),
               acc_norec=float(acc_norec), rate=float(S.mean()), **cert)
    print("RESULT", json.dumps(res), flush=True)
    name = f"sw2_{TAG}_s{SEED}"
    if H != 512:
        name += f"_h{H}"
    res["H"] = H
    json.dump(res, open(f"../results/{name}.json", "w"), indent=1)
    if os.environ.get("SAVE"):
        torch.save(m.state_dict(), os.path.expanduser(os.environ["SAVE"]))


if __name__ == "__main__":
    main()
