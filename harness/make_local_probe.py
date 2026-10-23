"""SPEED-005 probe: a SYNTHETIC locally connected network made from the trained dense R1 export by zeroing every
recurrent weight between 32-neuron groups more than one group apart on a ring of 16 (fan-in 96, the LOCAL_R=1
pattern of tab:scale). Not retrained: its accuracy is not claimed. It isolates the effect of bounded fan-in on
the execution engine with real certificates. Usage: python make_local_probe.py <data_dir>"""
import sys, numpy as np
d = sys.argv[1]
dims = open(f"{d}/cert_dims.txt").read().split()
P, C, T, N = map(int, dims[:4]); H = P * C; ND = int(dims[8])
g = np.arange(H) // 32; G = H // 32
dist = np.abs(g[:, None] - g[None, :]); dist = np.minimum(dist, G - dist)
mask = (dist <= 1).astype(np.float64)
for di in range(ND):
    W = np.fromfile(f"{d}/cert_W{di}.bin", dtype=np.float64).reshape(H, H)
    (W * mask).tofile(f"{d}/certloc_W{di}.bin")
    print(f"tap {di}: R mean dense {np.maximum(W,0).sum(1).mean():.3f} -> local {np.maximum(W*mask,0).sum(1).mean():.3f}")
import shutil
shutil.copy(f"{d}/cert_I.bin", f"{d}/certloc_I.bin"); shutil.copy(f"{d}/cert_dims.txt", f"{d}/certloc_dims.txt")
print("fan-in", int(mask.sum(1)[0]), "of", H)
