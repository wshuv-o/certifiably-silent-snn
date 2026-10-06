"""Failure analysis: WHERE does the certificate constraint cost accuracy, and is the loss necessary?

Key suspicion: the trained bound is R_i = sum_j relu(W_ij) over ALL presynaptic neurons, i.e. it assumes
every neuron fires simultaneously. Measured firing rates are 2.7-4.6%, so the bound may be ~20-35x more
pessimistic than the network's actual behaviour. If so, the accuracy cost is paid to satisfy an
unnecessarily loose bound, not because certifiable silence is intrinsically expensive.

Runs on CPU deliberately, to avoid contending with GPU jobs.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("H", "512")
from pilot_silence import fetch, T, NIN, NOUT

H = 512; CPC = 32; NPC = H // CPC          # 32 neurons per core, 16 cores
BETA = float(np.exp(-0.5)); THETA = 1.0
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02
BUDGET = (1 - BETA) * THETA
dev = "cpu"; torch.set_num_threads(8)
NS = int(os.environ.get("NS", "300"))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.win = torch.nn.Linear(NIN, H, bias=False)
        self.wrec = torch.nn.Linear(H, H, bias=False)
        self.wout = torch.nn.Linear(H, NOUT, bias=False)

    @torch.no_grad()
    def run(self, x):
        B = x.shape[0]
        v = torch.zeros(B, H); s = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(B, NOUT); iext = self.win(x); out = 0; S = []
        for t in range(T):
            a = RHO * a + GAMMA * s
            v = BETA * v + iext[:, t] + torch.nn.functional.linear(s, self.wrec.weight)
            s = ((v - (THETA + a)) >= 0).float()
            v = v - s * (THETA + a)
            u = BETA * u + self.wout(s); out = out + u
            S.append(s)
        return out, torch.stack(S, 1)


def load(p):
    m = Net(); sd = torch.load(os.path.expanduser(p), map_location="cpu")
    m.load_state_dict({k: v for k, v in sd.items() if k in ("win.weight", "wrec.weight", "wout.weight")})
    return m.eval()


def rk_bound(Wp, k):
    """Sound drive bound when at most k neurons per presynaptic CORE can fire in a step:
    for each target i and each core p, take the k largest positive weights from p. Sum over cores."""
    Wc = Wp.reshape(H, NPC, CPC)                       # [target, core, neuron-in-core]
    kk = min(k, CPC)
    return Wc.topk(kk, dim=2).values.sum(dim=(1, 2))    # [H]


Xte, yte = fetch("test")
idx = np.random.default_rng(0).choice(len(yte), NS, replace=False)
x = torch.tensor(Xte[idx], dtype=torch.float32)
y = yte[idx]

print(f"budget (1-beta)*theta = {BUDGET:.4f}   cores={NPC} x {CPC} neurons   n_samples={NS}\n")
out = {}
for name, path in (("control", "~/research/models/c2_ctrl_s2.pt"),
                   ("constrained (FT+cert0.3)", "~/research/models/c2_ours_s2.pt")):
    m = load(path)
    logits, S = m.run(x)
    acc = (logits.argmax(1).numpy() == y).mean()
    W = m.wrec.weight.data; Wp = torch.relu(W)
    R_full = Wp.sum(1)
    # measured simultaneous activity per core per step
    per_core = S.reshape(NS, T, NPC, CPC).sum(-1)       # active count per core per step
    rate = S.mean().item()
    mx = int(per_core.max().item())
    p999 = int(np.percentile(per_core.numpy(), 99.9))
    p100 = mx
    print(f"--- {name} ---")
    print(f"  accuracy {100*acc:.2f}%   firing rate {100*rate:.2f}%")
    print(f"  simultaneous active per core (of {CPC}): mean {per_core.mean():.2f}  "
          f"p99.9 {p999}  MAX {mx}")
    print(f"  R_full (all-fire bound):  mean {R_full.mean():.3f}   median {R_full.median():.3f}"
          f"   frac under budget {100*(R_full<=BUDGET).float().mean():.1f}%")
    for k in (1, 2, 4, p999, mx):
        Rk = rk_bound(Wp, k)
        print(f"  R_k  k={k:<3} (<= {k}/core):     mean {Rk.mean():.3f}"
              f"   tightening {R_full.mean()/Rk.mean():.1f}x"
              f"   frac under budget {100*(Rk<=BUDGET).float().mean():.1f}%")
    pos = Wp.sum().item(); neg = torch.relu(-W).sum().item()
    print(f"  weight mass: positive {pos:.1f}  negative {neg:.1f}  E/I ratio {pos/neg:.3f}")
    out[name] = dict(acc=float(acc), rate=rate, max_per_core=mx, p999=p999,
                     R_full=float(R_full.mean()), ei=pos/neg)
    print()
json.dump(out, open("../results/analysis_accuracy_loss.json", "w"), indent=1)
print("saved ../results/analysis_accuracy_loss.json")
