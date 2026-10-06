"""S2 (pre-registered in research/N3_SCALEUP_PLAN.md): certified-silence training on a stronger
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
BETA = float(np.exp(-0.5)); THETA = 1.0
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02                     # adaptation decay / jump
LAM = float(os.environ.get("CERT_LAMBDA", "0")); SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "40")); K = 4
DATASET = os.environ.get("DATASET", "shd")                     # shd | ssc (SSC-001)
ALT = os.environ.get("ALT", "none")                             # ALT-001: none | clamp | l1
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

    def forward(self, x):
        Bsz = x.shape[0]
        v = torch.zeros(Bsz, H, device=x.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(Bsz, NOUT, device=x.device)
        iext = self.win(x); out = 0; V, S = [], []
        for t in range(T):
            a = RHO * a + GAMMA * s
            v = BETA * v + iext[:, t] + self.wrec(s)
            thr = THETA + a
            s = Spike.apply(v - thr)
            v = v - s * thr
            u = BETA * u + self.wout(s); out = out + u
            V.append(v); S.append(s)
        return out, iext, torch.stack(V, 1), torch.stack(S, 1)


def cert_penalty(m, V, S, iext):
    R = torch.relu(m.wrec.weight).sum(1)
    tmax = T - K; silent = torch.ones_like(V[:, :tmax])
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k].detach())
        reach = BETA * reach + iext[:, k:tmax + k] + R
        worst = torch.maximum(worst, reach)
    return (torch.relu(worst - THETA) * silent).mean()


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
    W = m.wrec.weight; Wp = torch.relu(W)
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
        res["core_cert"] += cs.float().mean().item() * Bsz; res["core_oracle"] += co.float().mean().item() * Bsz
        res["neuron_cert"] += (~Sset).float().mean().item() * Bsz
        res["violations"] += int((~Sset & actual).sum().item()); n += Bsz
    for k in ("core_cert", "core_oracle", "neuron_cert"): res[k] /= n
    res["R_mean"] = float(Wp.sum(1).mean()); res["budget"] = (1 - BETA) * THETA
    return res


def main():
    if DATASET == "ssc":
        Xtr, ytr = fetch_ssc("train"); Xte, yte = fetch_ssc("test")
    else:
        Xtr, ytr = fetch("train"); Xte, yte = fetch("test")
    m = ALIFNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
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
            if LAM > 0:
                loss = loss + LAM * cert_penalty(m, V, S, iext)
            if ALT == "l1":                                      # ALT-001: L1 on excitatory drive
                loss = loss + 0.1 * torch.relu(m.wrec.weight).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ALT == "clamp":                                   # ALT-001: hard budget R_i <= 0.33
                with torch.no_grad():
                    W = m.wrec.weight; R = torch.relu(W).sum(1)
                    scale = torch.clamp(0.33 / R.clamp_min(1e-12), max=1.0)
                    W.copy_(torch.where(W > 0, W * scale[:, None], W))
        if ep % 5 == 4 or ep == EPOCHS - 1:
            m.eval(); print(f"epoch {ep} test acc {evaluate(m, Xte, yte):.4f} ({time.time() - t0:.0f}s/epoch)", flush=True)
    m.eval()
    acc = evaluate(m, Xte, yte)
    sub = np.random.default_rng(1).choice(len(yte), 500, replace=False)
    cert = certify(m, batch(Xte, sub))
    w0 = m.wrec.weight.data.clone(); m.wrec.weight.data.zero_(); acc_norec = evaluate(m, Xte, yte); m.wrec.weight.data.copy_(w0)
    _, _, _, S = m(torch.tensor(batch(Xte, sub[:100]), dtype=torch.float32, device=dev))
    res = dict(dataset=DATASET, alt=ALT, lam=LAM, seed=SEED, epochs=EPOCHS, acc=float(acc),
               acc_norec=float(acc_norec), rate=float(S.mean()), **cert)
    print("RESULT", json.dumps(res), flush=True)
    name = f"s2_l{LAM:g}_s{SEED}" if (DATASET == "shd" and ALT == "none") else f"s2_{DATASET}_{ALT}_l{LAM:g}_s{SEED}"
    if H != 512:
        name += f"_h{H}"
    res["H"] = H
    json.dump(res, open(f"../results/{name}.json", "w"), indent=1)
    if os.environ.get("SAVE"):
        torch.save(m.state_dict(), os.path.expanduser(os.environ["SAVE"]))


if __name__ == "__main__":
    main()
