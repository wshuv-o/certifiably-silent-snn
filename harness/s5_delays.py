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
RATE_L = float(os.environ.get("RATE_L", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "40")); K = int(os.environ.get("K", "4"))
LOCAL = os.environ.get("LOCAL", "0") == "1"; LOCAL_R = int(os.environ.get("LOCAL_R", "1"))
AUG = int(os.environ.get("AUG", "1"))
TEST = os.environ.get("TEST", "0") == "1"; TAG = os.environ.get("TAG", "cfg")
VALSPK = [int(s) for s in os.environ.get("VALSPK", "3,6").split(",")]
INIT = os.environ.get("INIT", "")
LR = float(os.environ.get("LR", "2e-3")); RAMP = os.environ.get("RAMP", "0") == "1"
# --- ACC-001 recipe knobs, from best_config_SHD.py (Hammouamri et al. 2024). Every default below is
# --- the pre-ACC-001 behaviour, so earlier results reproduce from this same file.
DROP = float(os.environ.get("DROP", "0"))        # reference 0.4; on the feedforward current only
BN = os.environ.get("BN", "0") == "1"            # reference True; feedforward current only (see note)
ONECYCLE = os.environ.get("ONECYCLE", "0") == "1"   # reference one_cycle; we had cosine
BS = int(os.environ.get("BS", "128"))            # reference 256
SURR = os.environ.get("SURR", "fs")              # fs = fast sigmoid squared (ours) | atan (reference)
ATAN_A = float(os.environ.get("ATAN_A", "5.0"))  # reference alpha
NBINS = int(os.environ.get("NBINS", "1"))        # reference 5: sum groups of adjacent input channels
FT_EP = int(os.environ.get("FT_EP", "0"))        # reference +30 epochs at a tenth of the LR
# --- ACC-002: per-neuron learnable membrane time constant. tau_i is confined to [TAU_MIN, TAU_MAX]
# --- by a sigmoid, so beta_i = exp(-1/tau_i) stays strictly inside (0,1) and every neuron keeps a
# --- non-zero excitatory budget (1 - beta_i) theta. Default off; off reproduces the scalar model.
TAU_LEARN = os.environ.get("TAU_LEARN", "0") == "1"
# Time constants move on a different scale from weights. The delay learner in s7_dcls.py failed for
# exactly this reason -- one learning rate for positions and weights -- so it is a knob here too
# rather than an assumption. 1.0 reproduces the single-rate behaviour.
LR_TAU_MULT = float(os.environ.get("LR_TAU_MULT", "1"))
TAU_MIN = float(os.environ.get("TAU_MIN", "1.0"))
TAU_MAX = float(os.environ.get("TAU_MAX", "8.0"))
# --- ARCH-001: a FEEDFORWARD front-end in front of the certified recurrent layer. Feedforward units
# --- depend only on the input, so they never wait on a neighbour and need no certificate; their
# --- output reaches the recurrent layer as its external current iext, which the certificate already
# --- treats as exactly known. The certificate code is therefore unchanged. Default "none" reproduces
# --- every earlier result.
#     ffd   : delayed feedforward taps straight into the recurrent layer, iext = sum_d W_d x(t-d)
#     layer : a feedforward LIF layer of FE_H units with delayed taps, whose spikes drive the recurrent layer
#     dcls  : ACC-003. The published SHD delay architecture (Hammouamri et al. 2024) as a front-end:
#             FE_LAYERS feedforward layers of FE_H near-memoryless LIF units, each synapse with a
#             learnable delay in [0, FE_DMAX-1] realised by a Gaussian kernel whose width anneals over
#             the first quarter of training, positions at FE_LR_POS_MULT x the weight rate, batch
#             normalisation and dropout on the front-end only, arctangent surrogate.
FE = os.environ.get("FE", "none")
assert FE in ("none", "ffd", "layer", "dcls"), f"unknown FE={FE}"
FE_LAYERS = int(os.environ.get("FE_LAYERS", "2"))
FE_DMAX = int(os.environ.get("FE_DMAX", "25"))              # reference: 250 ms at 10 ms steps
FE_LR_POS_MULT = float(os.environ.get("FE_LR_POS_MULT", "100"))
FE_SIG_END = float(os.environ.get("FE_SIG_END", "0.5"))
FE_DELAYS = sorted({int(d) for d in os.environ.get("FE_DELAYS", "0").split(",")})
assert min(FE_DELAYS) >= 0, "feedforward delays must be >= 0"
FE_H = int(os.environ.get("FE_H", "256"))
FE_TAU = float(os.environ.get("FE_TAU", "2.0")); BETA_FE = float(np.exp(-1.0 / FE_TAU))
FE_DROP = float(os.environ.get("FE_DROP", "0"))   # dropout on the front-end spikes (training only)
assert 0 < TAU_MIN < TAU_MAX, "need 0 < TAU_MIN < TAU_MAX"
assert TAU_MIN <= TAUM <= TAU_MAX, f"TAUM={TAUM} must lie inside [{TAU_MIN}, {TAU_MAX}] to initialise at it"
assert NIN % NBINS == 0, f"NIN={NIN} not divisible by NBINS={NBINS}"
NIN_EFF = NIN // NBINS
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
        x, = ctx.saved_tensors
        if SURR == "atan":
            # arctan surrogate, d/dx (1/pi) atan(pi/2 * a * x), as used by the reference via SpikingJelly
            return g * (ATAN_A / 2.0) / (1.0 + (np.pi / 2.0 * ATAN_A * x) ** 2)
        return g / (1 + 10 * x.abs()) ** 2


class SpikeAtan(torch.autograd.Function):
    """Heaviside forward, arctangent surrogate backward (alpha = 5), for the ACC-003 front-end only."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * (5.0 / 2.0) / (1.0 + (np.pi / 2.0 * 5.0 * x) ** 2)


class DCLSFrontEnd(torch.nn.Module):
    """ACC-003: feedforward layers with a learnable delay per synapse (Hammouamri et al. 2024).

    Each synapse (o, i) carries a weight W[o,i] and a continuous delay position P[o,i] in [0, FE_DMAX-1].
    Its kernel over integer delays k is W[o,i] * g_k, with g a normalised Gaussian centred at P whose
    width `sig` is annealed by the training loop. The layer output is a causal convolution over time,
    so it depends on the input only -- never on recurrent state.
    """

    def __init__(self, nin):
        super().__init__()
        self.w = torch.nn.ParameterList(); self.pos = torch.nn.ParameterList()
        self.bn = torch.nn.ModuleList()
        dims = [nin] + [FE_H] * FE_LAYERS
        for a, b in zip(dims[:-1], dims[1:]):
            w = torch.empty(b, a); torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))
            self.w.append(torch.nn.Parameter(w))
            self.pos.append(torch.nn.Parameter(torch.rand(b, a) * (FE_DMAX - 1)))
            self.bn.append(torch.nn.BatchNorm1d(b))
        self.drop = torch.nn.Dropout(FE_DROP) if FE_DROP > 0 else None
        self.register_buffer("sig", torch.tensor(float(FE_DMAX // 2)))
        self.register_buffer("taps", torch.arange(FE_DMAX, dtype=torch.float32))
        self.beta_fe = 1.0 - 1.0 / 1.005                        # reference LIF, tau = 10.05 ms at 10 ms steps

    def kernel(self, li):
        g = torch.exp(-0.5 * ((self.taps - self.pos[li].unsqueeze(-1)) / self.sig) ** 2)
        g = g / (g.sum(-1, keepdim=True) + 1e-9)
        return self.w[li].unsqueeze(-1) * g                    # (out, in, FE_DMAX), index k = delay k

    def clamp_(self):
        with torch.no_grad():
            for p in self.pos:
                p.clamp_(0, FE_DMAX - 1)

    def forward(self, x):                                      # x: (B, T, C)
        z = x.transpose(1, 2)                                  # (B, C, T)
        for li in range(len(self.w)):
            k = self.kernel(li).flip(-1)                       # conv1d correlates; flip so tap k = delay k
            cur = torch.nn.functional.conv1d(torch.nn.functional.pad(z, (FE_DMAX - 1, 0)), k)
            cur = self.bn[li](cur)                             # (B, H, T)
            v = torch.zeros(cur.shape[0], cur.shape[1], device=cur.device); out = []
            for t in range(cur.shape[2]):
                v = self.beta_fe * v + (1.0 - self.beta_fe) * cur[:, :, t]
                s = SpikeAtan.apply(v - THETA)
                v = v * (1.0 - s.detach())                     # hard reset, detached as in the reference
                out.append(s)
            z = torch.stack(out, 2)
            if self.drop is not None:
                z = self.drop(z)
        return z.transpose(1, 2)                               # (B, T, FE_H)


class DelayNet(torch.nn.Module):
    """One recurrent layer whose recurrence is a sum of delayed taps: sum_d W_d s(t-d)."""

    def __init__(self):
        super().__init__()
        # init order matches s4_improve.py (win, wrec, wout) so DELAYS=1 is as close to it as possible.
        # Exact bitwise reproduction is NOT expected: nn.Linear's default init consumes RNG that a raw
        # Parameter does not, so the equivalence check below is statistical, not bit-for-bit.
        if FE == "none":
            self.win = torch.nn.Linear(NIN_EFF, H, bias=False)
            torch.nn.init.normal_(self.win.weight, 0, 0.05)
        elif FE == "dcls":
            self.fe = DCLSFrontEnd(NIN_EFF)
            self.win = torch.nn.Linear(FE_H, H, bias=False)
            torch.nn.init.normal_(self.win.weight, 0, 0.05 * np.sqrt(NIN_EFF / FE_H))
        elif FE == "ffd":
            # one input matrix per feedforward delay; variance split so the initial current matches 1 tap
            self.fe_taps = torch.nn.ModuleList(
                [torch.nn.Linear(NIN_EFF, H, bias=False) for _ in FE_DELAYS])
            for lin in self.fe_taps:
                torch.nn.init.normal_(lin.weight, 0, 0.05 / np.sqrt(len(FE_DELAYS)))
        else:  # layer
            self.fe_taps = torch.nn.ModuleList(
                [torch.nn.Linear(NIN_EFF, FE_H, bias=False) for _ in FE_DELAYS])
            for lin in self.fe_taps:
                torch.nn.init.normal_(lin.weight, 0, 0.05 / np.sqrt(len(FE_DELAYS)))
            self.fe_bn = torch.nn.BatchNorm1d(FE_H)
            self.fe_drop = torch.nn.Dropout(FE_DROP) if FE_DROP > 0 else None
            self.win = torch.nn.Linear(FE_H, H, bias=False)
            torch.nn.init.normal_(self.win.weight, 0, 0.05 * np.sqrt(NIN_EFF / FE_H))
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
        # Normalisation and dropout act on the FEEDFORWARD current only. The certificate bounds the
        # recurrent drive by sum_j max(0, W_ij), which holds only because that drive is linear in the
        # spikes; normalising it would rescale it by gamma/sigma and break the bound. The feedforward
        # current is exactly known to the certificate, so transforming it costs nothing.
        if TAU_LEARN:
            # initialise every neuron at TAUM, with a small spread so they are not degenerate
            p0 = min(max((TAUM - TAU_MIN) / (TAU_MAX - TAU_MIN), 1e-3), 1 - 1e-3)
            self.traw = torch.nn.Parameter(
                torch.full((H,), float(np.log(p0 / (1 - p0)))) + 0.01 * torch.randn(H))
        self.bn = torch.nn.BatchNorm1d(H) if BN else None
        self.drop = torch.nn.Dropout(DROP) if DROP > 0 else None

    def masked(self):
        return [w * self.mask for w in self.wrec]

    @staticmethod
    def _delayed(x, d):
        """x(t - d) along the time axis, zero before the start."""
        if d == 0:
            return x
        return torch.cat([torch.zeros_like(x[:, :d]), x[:, :-d]], 1)

    def external_current(self, x):
        """The recurrent layer's external current. Depends on the input only, never on recurrent state."""
        if FE == "none":
            return self.win(x)
        if FE == "dcls":
            Z = self.fe(x)
            self.fe_rate = float(Z.detach().mean())
            return self.win(Z)
        cur = 0
        for lin, d in zip(self.fe_taps, FE_DELAYS):
            cur = cur + lin(self._delayed(x, d))
        if FE == "ffd":
            return cur
        Bsz = x.shape[0]
        cur = self.fe_bn(cur.reshape(-1, FE_H)).reshape(Bsz, -1, FE_H)
        vf = torch.zeros(Bsz, FE_H, device=x.device); Z = []
        for t in range(cur.shape[1]):
            vf = BETA_FE * vf + cur[:, t]
            z = Spike.apply(vf - THETA)
            vf = vf - z * THETA
            Z.append(z)
        Z = torch.stack(Z, 1)
        if self.fe_drop is not None:
            Z = self.fe_drop(Z)
        self.fe_rate = float(Z.detach().mean())
        return self.win(Z)

    def beta(self):
        """Membrane decay, per neuron when learned and scalar otherwise.

        Returned as a tensor of shape (H,) which broadcasts against v (B,H) and reach (B,tmax,H),
        so the certificate arithmetic is unchanged apart from the budget becoming neuron-specific.
        """
        if not TAU_LEARN:
            return BETA
        tau = TAU_MIN + (TAU_MAX - TAU_MIN) * torch.sigmoid(self.traw)
        return torch.exp(-1.0 / tau)

    def forward(self, x):
        Bsz = x.shape[0]
        v = torch.zeros(Bsz, H, device=x.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(Bsz, NOUT, device=x.device)
        if NBINS > 1:
            # sum groups of adjacent input channels, as the reference's n_bins does
            x = x.reshape(Bsz, x.shape[1], NIN_EFF, NBINS).sum(-1)
        iext = self.external_current(x)
        if self.bn is not None:
            # normalise each hidden unit's current over the batch and over time together
            iext = self.bn(iext.reshape(-1, H)).reshape(Bsz, -1, H)
        if self.drop is not None:
            iext = self.drop(iext)
        out = 0; V, S = [], []
        hist = [torch.zeros_like(v) for _ in range(DMAX)]     # hist[j] = s(t-1-j)
        Wm = self.masked()
        bet = self.beta()
        for t in range(T):
            a = RHO * a + GAMMA * s
            rec = 0
            for wi, d in enumerate(DELAYS):
                rec = rec + torch.nn.functional.linear(hist[d - 1], Wm[wi])
            v = bet * v + iext[:, t] + rec
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
    bet = m.beta()
    Bsz = V.shape[0]; tmax = T - K
    ones = torch.ones(Bsz, tmax, H, device=V.device)
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9)
    silent = torch.ones_like(reach)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k].detach())
        exact, bounded = delay_terms(S.detach(), Wm, Wp, ones, k, tmax, Bsz)
        reach = bet * reach + iext[:, k:tmax + k] + exact + bounded
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
    m.eval(); c = 0
    for i in range(0, len(y), 256):
        xb = torch.tensor(batch(X, range(i, min(i + 256, len(y)))), dtype=torch.float32, device=dev)
        c += (m(xb)[0].argmax(1).cpu().numpy() == y[i:i + 256]).sum()
    return c / len(y)


@torch.no_grad()
def certify(m, X):
    """Lemma-2 fixed-point certificate, generalised to delayed synapses."""
    m.eval()
    Wm = m.masked(); Wp = [torch.relu(w) for w in Wm]
    bet = m.beta()
    res = {"core_cert": 0.0, "core_oracle": 0.0, "violations": 0, "neuron_cert": 0.0, "allcore_cert": 0.0}
    n = 0
    for i in range(0, len(X), 100):
        _, iext, V, S = m(torch.tensor(batch(X, range(i, min(i + 100, len(X)))), dtype=torch.float32, device=dev))
        Bsz = V.shape[0]; tmax = T - K

        def may_fire(Sset_f):
            v = V[:, :tmax].clone(); fire = torch.zeros_like(v, dtype=torch.bool)
            for k in range(1, K + 1):
                exact, bounded = delay_terms(S, Wm, Wp, Sset_f, k, tmax, Bsz)
                v = bet * v + iext[:, k:tmax + k] + exact + bounded
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
    if TAU_LEARN:
        # the budget is now neuron-specific; report its distribution alongside the mean so a network
        # that buys excitation by leaking faster is visible rather than averaged away
        bud = (1 - bet) * THETA
        res["budget"] = float(bud.mean())
        res["budget_min"] = float(bud.min()); res["budget_max"] = float(bud.max())
        tau = -1.0 / torch.log(bet)
        res["tau_mean"] = float(tau.mean()); res["tau_min"] = float(tau.min())
        res["tau_max"] = float(tau.max()); res["tau_std"] = float(tau.std())
    else:
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
        if os.environ.get("INIT_ZERO_NEW"):                     # taps absent from INIT start at zero
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n.startswith("wrec.") and n not in sd:
                        p.zero_(); print("INIT zero:", n, flush=True)
    if TAU_LEARN and LR_TAU_MULT != 1.0:
        tp = [p for n, p in m.named_parameters() if n == "traw"]
        rest = [p for n, p in m.named_parameters() if n != "traw"]
        opt = torch.optim.AdamW([{"params": rest, "lr": LR},
                                 {"params": tp, "lr": LR * LR_TAU_MULT}], LR, weight_decay=1e-4)
        print("tau learning rate %g (%gx weights)" % (LR * LR_TAU_MULT, LR_TAU_MULT), flush=True)
    elif FE == "dcls":
        pos = [p for n, p in m.named_parameters() if n.startswith("fe.pos")]
        rest = [p for n, p in m.named_parameters() if not n.startswith("fe.pos")]
        opt = torch.optim.AdamW([{"params": rest, "lr": LR, "weight_decay": 1e-4},
                                 {"params": pos, "lr": LR * FE_LR_POS_MULT, "weight_decay": 0.0}], LR)
        print("delay positions at lr %g (%gx weights)" % (LR * FE_LR_POS_MULT, FE_LR_POS_MULT), flush=True)
    else:
        opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
    spe = max(1, (len(ytr) + BS - 1) // BS)
    steps = EPOCHS * spe
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(1, steps))
             if ONECYCLE else torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps)))
    t0 = time.time()
    def valacc():
        m.eval()
        if vsel is None:
            a = evaluate(m, Xva, yva); m.train(); return a
        with torch.no_grad():
            a = float((np.concatenate([m(torch.tensor(batch(Xva, vsel[i:i + 256]), dtype=torch.float32,
                      device=dev))[0].argmax(1).cpu().numpy() for i in range(0, len(vsel), 256)]) == yva).mean())
        m.train(); return a

    m.train()
    for ep in range(EPOCHS + FT_EP):
        if ep == EPOCHS and FT_EP:
            # fine-tuning tail at a tenth of the learning rate, constant, as the reference does
            for gp in opt.param_groups: gp["lr"] = LR / 10.0
            sched = None
            print(f"fine-tuning tail: {FT_EP} epochs at lr {LR / 10.0:g}", flush=True)
        if FE == "dcls":
            if INIT:
                # fine-tuning a trained front-end: its delays are already resolved, keep the final width
                m.fe.sig.fill_(FE_SIG_END)
            else:
                # reference schedule: kernel width decays exponentially from FE_DMAX//2 to FE_SIG_END over
                # the first quarter of training, then stays there (delays effectively integer from then on)
                q = max(1, EPOCHS // 4); s0 = float(FE_DMAX // 2)
                m.fe.sig.fill_(s0 * (FE_SIG_END / s0) ** min(1.0, ep / q))
        perm = tidx[np.random.permutation(len(tidx))]
        for b in range(0, len(perm), BS):
            idx = perm[b:b + BS]
            x = augment(torch.tensor(batch(Xtr, idx), dtype=torch.float32, device=dev))
            yb = torch.tensor(ytr[idx], device=dev)
            out, iext, V, S = m(x)
            loss = torch.nn.functional.cross_entropy(out, yb)
            if LAM > 0:
                lam_eff = LAM * (min(1.0, (ep + 1) / max(1, EPOCHS // 2)) if RAMP else 1.0)
                loss = loss + lam_eff * cert_penalty(m, V, S, iext)
            if RATE_L > 0:                                      # recurrent spike-rate penalty (longer silences)
                loss = loss + RATE_L * S.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if FE == "dcls":
                m.fe.clamp_()
            if sched is not None: sched.step()
        if ep % 5 == 4 or ep == EPOCHS + FT_EP - 1:
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
               recipe=dict(drop=DROP, bn=int(BN), onecycle=int(ONECYCLE), bs=BS, surr=SURR,
                           nbins=NBINS, ft_ep=FT_EP, lr=LR),
               arch=dict(fe=FE, fe_delays=FE_DELAYS, fe_h=FE_H, fe_tau=FE_TAU, fe_drop=FE_DROP,
                         fe_layers=FE_LAYERS, fe_dmax=FE_DMAX,
                         fe_pos_mean=(float(torch.cat([p.detach().flatten() for p in m.fe.pos]).mean())
                                      if FE == "dcls" else None),
                         n_params=int(sum(p.numel() for p in m.parameters()))),
               test=({'acc': float(acc), **{('test_' + k): v for k, v in cert_test.items()}} if TEST else None),
               **cert)
    print("RESULT", json.dumps(res), flush=True)
    name = f"d1_{TAG}_s{SEED}"
    json.dump(res, open(f"../results/{name}.json", "w"), indent=1)
    if os.environ.get("SAVE"):
        torch.save(m.state_dict(), os.path.expanduser(os.environ["SAVE"]))


if __name__ == "__main__":
    main()
