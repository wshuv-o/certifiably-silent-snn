"""Decisive test: what does recurrence actually CONTRIBUTE — excitation or inhibition?

The constraint only bounds POSITIVE recurrent weights. The analysis showed it satisfies the budget by
cutting positive weight mass 12x (E/I 0.650 -> 0.045) while growing inhibition. If the task barely needs
recurrent EXCITATION, that explains why certifiable silence is cheap at 512 -- and predicts it will be
expensive wherever recurrent excitation matters.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("H", "512")
from pilot_silence import fetch, T, NIN, NOUT
H = 512; BETA = float(np.exp(-0.5)); THETA = 1.0
RHO, GAMMA = float(np.exp(-14 / 200)), 0.02
torch.set_num_threads(8); NS = int(os.environ.get("NS", "400"))

class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.win = torch.nn.Linear(NIN, H, bias=False); s.wrec = torch.nn.Linear(H, H, bias=False)
        s.wout = torch.nn.Linear(H, NOUT, bias=False)
    @torch.no_grad()
    def run(s, x, Wrec):
        B = x.shape[0]; v = torch.zeros(B, H); sp = torch.zeros_like(v); a = torch.zeros_like(v)
        u = torch.zeros(B, NOUT); iext = s.win(x); out = 0; tot = 0.0
        for t in range(T):
            a = RHO * a + GAMMA * sp
            v = BETA * v + iext[:, t] + torch.nn.functional.linear(sp, Wrec)
            sp = ((v - (THETA + a)) >= 0).float(); v = v - sp * (THETA + a)
            u = BETA * u + s.wout(sp); out = out + u; tot += sp.mean().item()
        return out, tot / T

def load(p):
    m = Net(); sd = torch.load(os.path.expanduser(p), map_location="cpu")
    m.load_state_dict({k: v for k, v in sd.items() if k in ("win.weight","wrec.weight","wout.weight")})
    return m.eval()

Xte, yte = fetch("test")
idx = np.random.default_rng(0).choice(len(yte), NS, replace=False)
x = torch.tensor(Xte[idx], dtype=torch.float32); y = yte[idx]
res={}
for name, path in (("control","~/research/models/c2_ctrl_s2.pt"),
                   ("constrained","~/research/models/c2_ours_s2.pt")):
    m = load(path); W = m.wrec.weight.data.clone()
    variants = {
        "full recurrence":        W,
        "EXCITATION removed":     -torch.relu(-W),      # keep negatives only
        "INHIBITION removed":      torch.relu(W),       # keep positives only
        "no recurrence":           torch.zeros_like(W),
    }
    print(f"--- {name} ---")
    res[name]={}
    for vn, Wv in variants.items():
        out, rate = m.run(x, Wv)
        acc = 100*(out.argmax(1).numpy()==y).mean()
        print(f"   {vn:22s} acc {acc:6.2f}%   rate {100*rate:5.2f}%")
        res[name][vn]=acc
    base=res[name]["full recurrence"]
    print(f"   -> cost of removing excitation: {res[name]['EXCITATION removed']-base:+.2f} pts")
    print(f"   -> cost of removing inhibition: {res[name]['INHIBITION removed']-base:+.2f} pts")
    print(f"   -> cost of removing all:        {res[name]['no recurrence']-base:+.2f} pts\n")
json.dump(res, open("../results/analysis_ei_ablation.json","w"), indent=1)
print("saved ../results/analysis_ei_ablation.json")
