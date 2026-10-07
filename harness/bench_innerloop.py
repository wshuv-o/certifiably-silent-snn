"""Measure where the training step's time actually goes, and what the candidate fixes buy.

The inner loop is 100 sequential timesteps of small dependent kernels, which is the one shape a big
GPU cannot exploit. This times the real loop against two numerically-checked variants:

  base   : as in s5_delays.py -- one linear per delay tap
  fused  : the delay taps concatenated into a single (B, |D|H) @ (|D|H, H) matmul
  graph  : the whole forward+backward captured as a CUDA graph and replayed

Each variant is checked against `base` for agreement before it is timed, so a speed-up that changed
the arithmetic would be caught rather than reported.

    python harness/bench_innerloop.py
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

dev = "cuda"
H = int(os.environ.get("H", "512"))
NIN, NOUT, T = 700, 20, 100
DELAYS = [2, 4, 8]
DMAX = max(DELAYS)
B = 128
BETA, THETA, RHO, GAMMA = float(np.exp(-0.5)), 1.0, float(np.exp(-14 / 200)), 0.02
BETA_OUT = float(np.exp(-0.5))


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


class Net(torch.nn.Module):
    def __init__(self, fused):
        super().__init__()
        torch.manual_seed(0)
        self.win = torch.nn.Linear(NIN, H, bias=False)
        self.wrec = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(H, H).normal_(0, 0.02 / np.sqrt(len(DELAYS))))
             for _ in DELAYS])
        self.wout = torch.nn.Linear(H, NOUT, bias=False)
        self.register_buffer("mask", torch.ones(H, H))
        self.fused = fused

    def forward(self, x):
        v = torch.zeros(B, H, device=x.device)
        s = torch.zeros_like(v)
        a = torch.zeros_like(v)
        u = torch.zeros(B, NOUT, device=x.device)
        iext = self.win(x)
        out = 0
        hist = [torch.zeros_like(v) for _ in range(DMAX)]
        Wm = [w * self.mask for w in self.wrec]
        # one matmul instead of |D|: sum_d h_d W_d^T == cat(h_d) cat(W_d)^T
        Wcat = torch.cat(Wm, 1) if self.fused else None
        for t in range(T):
            a = RHO * a + GAMMA * s
            if self.fused:
                rec = F.linear(torch.cat([hist[d - 1] for d in DELAYS], 1), Wcat)
            else:
                rec = 0
                for wi, d in enumerate(DELAYS):
                    rec = rec + F.linear(hist[d - 1], Wm[wi])
            v = BETA * v + iext[:, t] + rec
            thr = THETA + a
            s = Spike.apply(v - thr)
            v = v - s * thr
            u = BETA_OUT * u + self.wout(s)
            out = out + u
            hist = [s] + hist[:-1]
        return out


def run(net, x, y):
    net.zero_grad(set_to_none=True)
    loss = F.cross_entropy(net(x), y)
    loss.backward()
    return loss


def timeit(fn, n=8):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


print("H=%d  B=%d  T=%d  delays=%s" % (H, B, T, DELAYS))
print("device: %s" % torch.cuda.get_device_name(0))

torch.manual_seed(1)
x = torch.rand(B, T, NIN, device=dev)
y = torch.randint(0, NOUT, (B,), device=dev)

base = Net(False).to(dev)
fused = Net(True).to(dev)
fused.load_state_dict(base.state_dict())

# ---- agreement before timing -----------------------------------------------------------------
lb = run(base, x, y)
lf = run(fused, x, y)
gb = torch.cat([p.grad.flatten() for p in base.parameters()])
gf = torch.cat([p.grad.flatten() for p in fused.parameters()])
print("\nloss  base %.8f  fused %.8f  |diff| %.2e" % (lb.item(), lf.item(), abs(lb.item() - lf.item())))
print("grad  max|diff| %.2e   rel %.2e" % ((gb - gf).abs().max().item(),
                                           ((gb - gf).norm() / gb.norm()).item()))

# ---- timing ------------------------------------------------------------------------------------
tb = timeit(lambda: run(base, x, y))
tf = timeit(lambda: run(fused, x, y))
print("\nper training step (fwd+bwd), batch of %d:" % B)
print("  base   %7.2f ms" % (tb * 1e3))
print("  fused  %7.2f ms   %.2fx" % (tf * 1e3, tb / tf))

# ---- forward-only, to size how much is backward -------------------------------------------------
with torch.no_grad():
    tfwd = timeit(lambda: base(x))
print("  forward only (no grad) %7.2f ms  -> backward is %.0f%% of the step"
      % (tfwd * 1e3, 100 * (1 - tfwd / tb)))

# ---- what an epoch costs, against the arithmetic ------------------------------------------------
nb = 53                                   # batches per epoch on SHD after the speaker-disjoint split
flop = 2 * B * T * (NIN * H + len(DELAYS) * H * H + H * NOUT) * 3   # fwd+bwd ~3x forward
print("\nepoch estimate: base %.1f s   fused %.1f s   (%d batches)" % (tb * nb, tf * nb, nb))
print("arithmetic per step %.1f GFLOP -> %.0f TFLOP/s achieved, card does ~56 TFLOP/s fp32"
      % (flop / 1e9, flop / tb / 1e12))

# ---- CUDA graph: capture forward+backward once, replay it ----------------------------------------
# Numerically this replays the identical kernels, so any divergence from base would be a bug.
print("\n--- CUDA graph capture ---")
try:
    gnet = Net(False).to(dev)
    gnet.load_state_dict(base.state_dict())
    sx, sy = x.clone(), y.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            gnet.zero_grad(set_to_none=False)
            F.cross_entropy(gnet(sx), sy).backward()
    torch.cuda.current_stream().wait_stream(side)

    for p in gnet.parameters():
        p.grad.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = F.cross_entropy(gnet(sx), sy)
        static_loss.backward()

    def graph_step():
        for p in gnet.parameters():
            p.grad.zero_()
        g.replay()

    graph_step()
    gg = torch.cat([p.grad.flatten() for p in gnet.parameters()])
    print("loss  base %.8f  graph %.8f" % (lb.item(), static_loss.item()))
    print("grad  max|diff| vs base %.2e" % (gb - gg).abs().max().item())
    tg = timeit(graph_step)
    print("  graph  %7.2f ms   %.2fx over base" % (tg * 1e3, tb / tg))
    print("epoch estimate: graph %.1f s" % (tg * nb))
except Exception as e:
    print("capture failed: %s: %s" % (type(e).__name__, str(e)[:200]))
