"""ACC-012: our certified recurrent delay layer inside the published SpikCommander SHD network.

SpikCommander (Wang et al. 2025) maps the input through its encoder (SEE) and spiking-transformer block to a
(T, B, 128) sequence, read out per time step by a linear head and trained on the time-summed softmax. ACC-012
inserts the recurrent ALIF layer of this paper (H = 512, 16 cores x 32, taps D = {2,4,8}, tau_m = 2 steps,
adaptation, fast-sigmoid surrogate, K = 4 certificate; identical to acc005_rec.RecLayer) between the block
output and the head, which now reads 512 recurrent spikes. The layer's external current is BN(W_in z) of the
block output, a function of the input only, so the certificate sees it exactly. Copied into
~/research/refs/SCommander-protocol at launch.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import layer
from spikcommder import SpikCommander

E = os.environ.get
REC_H = int(E("REC_H", "512")); CPC = 32; NPC = REC_H // CPC
DELAYS = sorted({int(d) for d in E("REC_DELAYS", "2,4,8").split(",")}); DMAX = max(DELAYS)
K = int(E("K", "4"))
# ACC-012c: adaptation-aware certificate. Before a neuron's first spike inside the window its adaptation only
# decays, so its threshold at t+k is exactly THETA + RHO^k a(t) + GAMMA RHO^(k-1) s(t), all measured at t.
CERT_ADAPT = E("CERT_ADAPT", "0") == "1"
BETA = float(np.exp(-1.0 / 2.0)); THETA = 1.0
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


class RecLayer(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.win = nn.Linear(nin, REC_H, bias=False)
        nn.init.normal_(self.win.weight, 0, 0.05 * np.sqrt(140 / nin))
        self.bn = nn.BatchNorm1d(REC_H)
        self.wrec = nn.ParameterList(
            [nn.Parameter(torch.randn(REC_H, REC_H) * (0.02 / np.sqrt(len(DELAYS)))) for _ in DELAYS])

    def forward(self, z):                                  # z: (T, B, nin)
        T, B = z.shape[:2]
        iext = self.bn(self.win(z).reshape(-1, REC_H)).reshape(T, B, REC_H)
        v = torch.zeros(B, REC_H, device=z.device); s = torch.zeros_like(v); a = torch.zeros_like(v)
        hist = [torch.zeros_like(v) for _ in range(DMAX)]
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
        self.V = torch.stack(V, 1); self.S = torch.stack(S, 1); self.A = torch.stack(A, 1)
        self.iext = iext.transpose(0, 1)
        return torch.stack(S, 0)


class SpikCommanderRec(SpikCommander):
    def __init__(self, config):
        super().__init__(config)                           # reference init, incl. its xavier re-init
        self.rec = RecLayer(config.n_hidden_neurons)       # created after it, keeps its own init
        self.head = layer.Linear(REC_H, config.n_outputs, bias=False, step_mode='m')
        xavier_uniform_(self.head.weight)                  # the reference's init for its head

    def forward(self, x, attention_mask):
        x = self.see(x)
        for module in self.blocks:
            x = module(x, attention_mask)
        x = self.rec(x)
        if self.config.use_dp:
            return self.head(self.final_dp(x))
        return self.head(x)


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


@torch.no_grad()
def certify(model, loader, device, mask_fn, reset_fn):
    model.eval()
    res = dict(core_cert=0.0, core_oracle=0.0, neuron_cert=0.0, violations=0); n = 0
    W = list(model.rec.wrec); Wp = [torch.relu(w) for w in W]
    for x, y, l in loader:
        model(x.float().to(device), mask_fn(l).transpose(0, 1).to(device))
        V, S, iext = model.rec.V, model.rec.S, model.rec.iext
        reset_fn(model)
        Bsz, T = V.shape[:2]; tmax = T - K

        A0, S0 = model.rec.A[:, :tmax], S[:, :tmax]

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
    res.update(R_per_delay=Rd, R_short=float(sum(r for d, r in zip(DELAYS, Rd) if d < K)),
               budget=(1 - BETA) * THETA, cert_adapt=CERT_ADAPT, pct_oracle=res["core_cert"] / max(res["core_oracle"], 1e-12))
    return res
