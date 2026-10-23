"""Export the certified recurrent layer of a state-of-the-art host network for the execution engine.

The engine simulates only the recurrent layer; its external current is computed here by the host network
(eval mode; DCLS with integer delays) on the first N test samples, padded to the batch's longest sequence.
Writes <tag>_W{0,1,2}.bin, <tag>_I.bin, <tag>_dims.txt in the s6 format to OUT. Run from the host's
protocol directory (SNN-delays-protocol for HOST=dcls, SCommander-protocol for HOST=scmd).
Env: HOST, CKPT, TAG, OUT, N (default 200)."""
import os, numpy as np, torch
E = os.environ.get
HOST, CKPT, TAG = E("HOST"), os.path.expanduser(E("CKPT")), E("TAG")
OUT = os.path.expanduser(E("OUT", "~/research/data/e6")); N = int(E("N", "200")); os.makedirs(OUT, exist_ok=True)
dev = "cuda"
if HOST == "dcls":
    from config import Config
    from protocol_data import loaders
    import acc005_rec as R
    c = Config; c.batch_size = N
    _, _, te = loaders(c)
    m = R.SnnDelaysRec(c).to(dev); m.load_state_dict(torch.load(CKPT, map_location=dev))
    x, y, l = next(iter(te))
    with torch.no_grad(), R.integer_delays(m):
        m(x.permute(1, 0, 2).float().to(dev)); rec = m.rec
        iext, W = rec.iext.double().cpu().numpy(), [w.detach().double().cpu().numpy() for w in rec.wrec]
        S = rec.S.cpu().numpy()
else:
    from best_config_SHD_former import Config
    from scmd_protocol_data import loaders, padded_sequence_mask
    import scmd_rec as R
    c = Config(); c.batch_size = N; c.backend = "torch"
    c.n_hidden_neurons = c.n_hidden_neurons_list[0]; c.hidden_dims = c.n_hidden_neurons
    _, _, te = loaders(c)
    m = R.SpikCommanderRec(c).to(dev); m.load_state_dict(torch.load(CKPT, map_location=dev)); m.eval()
    x, y, l = next(iter(te))
    with torch.no_grad():
        m(x.float().to(dev), padded_sequence_mask(l).transpose(0, 1).to(dev)); rec = m.rec
        iext, W = rec.iext.double().cpu().numpy(), [w.detach().double().cpu().numpy() for w in rec.wrec]
        S = rec.S.cpu().numpy()
Bn, T, H = iext.shape
for di, w in enumerate(W):
    np.ascontiguousarray(w).tofile(f"{OUT}/{TAG}_W{di}.bin")
np.ascontiguousarray(iext).tofile(f"{OUT}/{TAG}_I.bin")
P = 4; C = H // P
with open(f"{OUT}/{TAG}_dims.txt", "w") as f:
    f.write(f"{P} {C} {T} {Bn} {R.BETA!r} {R.THETA!r} {R.RHO!r} {R.GAMMA!r} {len(R.DELAYS)} "
            + " ".join(str(d) for d in R.DELAYS) + "\n")
Rd = [float(np.maximum(w, 0).sum(1).mean()) for w in W]
print(f"{TAG}: host={HOST} N={Bn} T={T} H={H} R_per_delay={[round(r, 3) for r in Rd]} model spike rate {S.mean():.4f}")
