"""ACC-005: our certified recurrent delay layer inside the published DCLS SHD network.

The network is the reference's (Hammouamri et al. 2024, best_config_SHD): two feedforward layers of 256 LIF
neurons with one learnable delay per synapse, then a learnable-delay readout. ACC-005 inserts the recurrent
ALIF layer of this paper (H = 512, 16 cores of 32, recurrent taps D = {2,4,8}, the same neuron, surrogate
and certificate as harness/s5_delays.py) between the second feedforward layer and the readout. The
recurrent layer's external current is a linear map (plus batch norm) of the second layer's spikes, which
depend on the input only, so the certificate sees it exactly; the readout is downstream of the recurrence
and never enters the certificate. Copied next to the reference code and imported from there.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer
from DCLS.construct.modules import Dcls1d
from snn_delays import SnnDelays

E = os.environ.get
REC_H = int(E("REC_H", "512")); CPC = 32; NPC = REC_H // CPC
DELAYS = sorted({int(d) for d in E("REC_DELAYS", "2,4,8").split(",")}); DMAX = max(DELAYS)
K = int(E("K", "4"))
LAM = float(E("CERT_LAMBDA", "0")); RAMP = E("RAMP", "1") == "1"
REC_DROP = float(E("REC_DROP", "0.4"))     # on the recurrent layer's spikes into the readout only
REC_BN = E("REC_BN", "1") == "1"           # on the external current only (see s5_delays.py note)
BETA = float(np.exp(-1.0 / 2.0)); THETA = 1.0     # TAUM = 2 steps, as in the paper
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02
# ACC-009: speaker-adversarial weight (gradient reversal on the recurrent layer's time-mean spikes). 0 = off,
# and then nothing below changes (no extra module, no extra RNG draw, same state dict).
ADV = float(E("ADV", "0")); N_SPK = 8
CERT_ADAPT = E("CERT_ADAPT", "0") == "1"   # ACC-012c adaptation-aware certificate (see scmd_rec.py)


class Spike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid-squared surrogate backward (s5_delays.py default)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


class RecLayer(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.win = nn.Linear(nin, REC_H, bias=False)
        nn.init.normal_(self.win.weight, 0, 0.05 * np.sqrt(140 / nin))
        self.bn = nn.BatchNorm1d(REC_H) if REC_BN else None
        self.wrec = nn.ParameterList(
            [nn.Parameter(torch.randn(REC_H, REC_H) * (0.02 / np.sqrt(len(DELAYS)))) for _ in DELAYS])

    def forward(self, z):                                  # z: (T, B, nin) spikes of the second layer
        T, B = z.shape[:2]
        iext = self.win(z)
        if self.bn is not None:
            iext = self.bn(iext.reshape(-1, REC_H)).reshape(T, B, REC_H)
        v = torch.zeros(B, REC_H, device=z.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        hist = [torch.zeros_like(v) for _ in range(DMAX)]  # hist[j] = s(t-1-j)
        V, S, A = [], [], []
        for t in range(T):
            a = RHO * a + GAMMA * s
            rec = 0
            for w, d in zip(self.wrec, DELAYS):
                rec = rec + F.linear(hist[d - 1], w)
            v = BETA * v + iext[t] + rec
            thr = THETA + a
            s = Spike.apply(v - thr)
            v = v - s * thr
            V.append(v); S.append(s); A.append(a)
            hist = [s] + hist[:-1]
        self.V = torch.stack(V, 1); self.S = torch.stack(S, 1)   # (B, T, H)
        self.A = torch.stack(A, 1)
        self.iext = iext.transpose(0, 1)                        # (B, T, H)
        return torch.stack(S, 0)                                # (T, B, H)


def delay_terms(S, W, Wp, Sset_f, k, tmax):
    Bsz = S.shape[0]
    Spad = torch.cat([torch.zeros(Bsz, DMAX, REC_H, device=S.device, dtype=S.dtype), S], 1)
    exact = 0; bounded = 0
    for wi, d in enumerate(DELAYS):
        if d >= k:
            j = DMAX + k - d
            exact = exact + Spad[:, j:j + tmax] @ W[wi].T
        else:
            bounded = bounded + Sset_f @ Wp[wi].T
    return exact, bounded


def cert_penalty(rec):
    V, S, iext = rec.V, rec.S.detach(), rec.iext
    W = list(rec.wrec); Wp = [torch.relu(w) for w in W]
    Bsz, T = V.shape[:2]; tmax = T - K
    ones = torch.ones(Bsz, tmax, REC_H, device=V.device)
    reach = V[:, :tmax]; worst = torch.full_like(reach, -1e9); silent = torch.ones_like(reach)
    for k in range(1, K + 1):
        silent = silent * (1 - S[:, k:tmax + k])
        exact, bounded = delay_terms(S, W, Wp, ones, k, tmax)
        reach = BETA * reach + iext[:, k:tmax + k] + exact + bounded
        worst = torch.maximum(worst, reach)
    return (torch.relu(worst - THETA) * silent).mean()


class SnnDelaysRec(SnnDelays):
    def build_model(self):
        super().build_model()
        c = self.config
        self.blocks[-1][0][0] = Dcls1d(REC_H, c.n_outputs, kernel_count=c.kernel_count, groups=1,
                                       dilated_kernel_size=c.max_delay, bias=c.bias, version=c.DCLSversion)
        self.model = nn.Sequential(*[l for block in self.blocks for sub in block for l in sub])
        self.rec = RecLayer(c.n_hidden_neurons)
        self.rec_drop = layer.Dropout(REC_DROP, step_mode='m')   # one mask over time, as the reference
        self.positions, self.weights, self.weights_bn, self.weights_plif = [], [], [], []
        for m in self.model.modules():
            if isinstance(m, Dcls1d):
                self.positions.append(m.P); self.weights.append(m.weight)
                if c.bias:
                    self.weights_bn.append(m.bias)
            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn += [m.weight, m.bias]
        self.weights += [self.rec.win.weight, *self.rec.wrec]
        if ADV > 0:
            self.adv = nn.Linear(REC_H, N_SPK)
            self.weights.append(self.adv.weight); self.weights_bn.append(self.adv.bias)
        if self.rec.bn is not None:
            self.weights_bn += [self.rec.bn.weight, self.rec.bn.bias]
        self._epoch = 0

    def decrease_sig(self, epoch):
        self._epoch = epoch + 1                     # called at the end of each epoch
        super().decrease_sig(epoch)

    def forward(self, x):                           # x: (T, B, C), as the reference
        c = self.config
        self._spk = None
        if x.shape[-1] == c.n_inputs + 1:              # ACC-009 training batches carry the speaker
            self._spk = x[0, :, -1].long(); x = x[..., :-1]
        for b in range(c.n_hidden_layers):
            x = x.permute(1, 2, 0)
            x = F.pad(x, (c.left_padding, c.right_padding), 'constant', 0)
            x = self.blocks[b][0][0](x)
            x = x.permute(2, 0, 1)
            if c.use_batchnorm:
                x = self.blocks[b][0][1](x.unsqueeze(3)).squeeze(3)
            x = self.blocks[b][1][1](self.blocks[b][1][0](x))
        x = self.rec_drop(self.rec(x))
        x = x.permute(1, 2, 0)
        x = F.pad(x, (c.left_padding, c.right_padding), 'constant', 0)
        out = self.blocks[-1][0][0](x).permute(2, 0, 1)
        self.blocks[-1][1][0](out)
        return self.blocks[-1][1][0].v_seq

    def calc_loss(self, output, y):
        loss = super().calc_loss(output, y)
        if ADV > 0 and self.training and self._spk is not None:
            z = GradReverse.apply(self.rec.S.mean(1))
            loss = loss + ADV * F.cross_entropy(self.adv(z), self._spk.to(z.device))
        if LAM > 0 and self.training:
            ramp = min(1.0, (self._epoch + 1) / max(1, self.config.epochs // 2)) if RAMP else 1.0
            loss = loss + LAM * ramp * cert_penalty(self.rec)
        return loss


class integer_delays:
    """The reference's evaluation transform (eval_model): sigma 0, 'max' kernels, rounded positions.
    Restores the training state on exit."""
    def __init__(self, m):
        self.m = m

    def __enter__(self):
        self.sd = {k: v.detach().clone() for k, v in self.m.state_dict().items()}
        self.m.eval()
        for blk in self.m.blocks:
            blk[0][0].SIG *= 0; blk[0][0].version = 'max'; blk[0][0].DCK.version = 'max'
        self.m.round_pos()
        return self.m

    def __exit__(self, *a):
        for blk in self.m.blocks:
            blk[0][0].version = 'gauss'; blk[0][0].DCK.version = 'gauss'
        self.m.load_state_dict(self.sd, strict=True)


@torch.no_grad()
def accuracy(m, loader, device):
    """Per-sample accuracy (the reference averages per batch), under integer delays."""
    c = n = 0
    with integer_delays(m):
        for x, y, _ in loader:
            out = m(x.permute(1, 0, 2).float().to(device))
            pred = torch.softmax(out, 2).sum(0).argmax(1).cpu()
            c += int((pred == y).sum()); n += len(y)
            m.reset_model(train=False)
    return c / n


@torch.no_grad()
def certify(m, loader, device):
    """Fixed-point K-step silence certificate of s5_delays.certify, on the recurrent layer."""
    res = dict(core_cert=0.0, core_oracle=0.0, neuron_cert=0.0, violations=0); n = 0
    with integer_delays(m):
        W = list(m.rec.wrec); Wp = [torch.relu(w) for w in W]
        for x, y, _ in loader:
            m(x.permute(1, 0, 2).float().to(device))
            V, S, iext = m.rec.V, m.rec.S, m.rec.iext
            m.reset_model(train=False)
            Bsz, T = V.shape[:2]; tmax = T - K

            A0, S0 = m.rec.A[:, :tmax], S[:, :tmax]

            def may_fire(Sset_f):
                v = V[:, :tmax].clone(); fire = torch.zeros_like(v, dtype=torch.bool)
                for k in range(1, K + 1):
                    exact, bounded = delay_terms(S, W, Wp, Sset_f, k, tmax)
                    v = BETA * v + iext[:, k:tmax + k] + exact + bounded
                    thr = THETA + ((RHO ** k) * A0 + GAMMA * (RHO ** (k - 1)) * S0 if CERT_ADAPT else 0.0)
                    fire |= v >= thr
                return fire

            Sset = torch.ones(Bsz, tmax, REC_H, dtype=torch.bool, device=device)
            for _ in range(50):
                new = may_fire(Sset.float()) & Sset
                if torch.equal(new, Sset):
                    break
                Sset = new
            actual = torch.zeros_like(Sset)
            for k in range(1, K + 1):
                actual |= S[:, k:tmax + k] > 0
            cs = (~Sset).reshape(Bsz, tmax, NPC, CPC).all(-1)
            co = (~actual).reshape(Bsz, tmax, NPC, CPC).all(-1)
            res["core_cert"] += cs.float().mean().item() * Bsz
            res["core_oracle"] += co.float().mean().item() * Bsz
            res["neuron_cert"] += (~Sset).float().mean().item() * Bsz
            res["violations"] += int((~Sset & actual).sum().item()); n += Bsz
        for k in ("core_cert", "core_oracle", "neuron_cert"):
            res[k] /= n
        Rd = [float(torch.relu(w).sum(1).mean()) for w in W]
    res["R_per_delay"] = Rd
    res["R_short"] = float(sum(r for d, r in zip(DELAYS, Rd) if d < K))
    res["budget"] = (1 - BETA) * THETA
    res["pct_oracle"] = res["core_cert"] / max(res["core_oracle"], 1e-12)
    return res


@torch.no_grad()
def per_class(m, loader, device, n=20):
    """Per-class accuracy under integer delays (WARP-DIAG: class 19 carries 40% of validation errors)."""
    hit = np.zeros(n); tot = np.zeros(n)
    with integer_delays(m):
        for x, y, _ in loader:
            pred = torch.softmax(m(x.permute(1, 0, 2).float().to(device)), 2).sum(0).argmax(1).cpu().numpy()
            m.reset_model(train=False)
            for t, p in zip(y.numpy(), pred):
                tot[t] += 1; hit[t] += (t == p)
    return [float(h / max(1, t)) for h, t in zip(hit, tot)]
