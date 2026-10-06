"""EXP-005 prep: graphs with controlled dependency depth for the asynchrony test.

Synthetic: N = 2^20, F = 8, locality window in {16, 256, 4096, random}; a small
window gives a long diameter (many tiny BFS rounds), random gives a short diameter.
Real: SNAP roadNet-CA (high diameter, low degree).
Writes weighted edge lists (integer weights 1..100 for SSSP), converts to Galois
.gr, and records BFS depth (rounds) and mean frontier fraction from source 0.
"""
import gzip, json, os, subprocess, urllib.request
import numpy as np
import pandas as pd
from graphgen import make_graph

DATA = os.path.expanduser("~/research/data/exp005")
CONV = os.path.expanduser("~/research/refs/Galois/build/tools/graph-convert/graph-convert")
os.makedirs(DATA, exist_ok=True)


def bfs_profile(N, src, dst):
    """Rounds and mean frontier fraction of BFS from vertex 0 (out-edges)."""
    order = np.argsort(src, kind="stable")
    s, d = src[order], dst[order]
    ptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(s, minlength=N), out=ptr[1:])
    seen = np.zeros(N, bool); seen[0] = True
    front = np.array([0]); sizes = []
    while front.size:
        sizes.append(front.size)
        starts = ptr[front]
        lens = ptr[front + 1] - starts
        # edge indices of all frontier vertices: starts[i] + 0..lens[i]-1
        idx = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(lens.sum())
        nb = np.unique(d[idx])
        nb = nb[~seen[nb]]
        seen[nb] = True
        front = nb
    return dict(rounds=len(sizes), mean_frontier_frac=float(np.mean(sizes) / N),
                reached_frac=float(seen.mean()))


def write_and_convert(name, N, src, dst, rng):
    gr = f"{DATA}/{name}.gr"
    if not os.path.exists(gr):
        el = f"{DATA}/{name}.el"
        wts = rng.integers(1, 101, size=src.size)
        pd.DataFrame({"s": src, "d": dst, "w": wts}).to_csv(el, sep=" ", header=False, index=False)
        subprocess.run([CONV, "--edgelist2gr", "--edgeType=uint32", el, gr], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(el)
    prof = bfs_profile(N, src, dst)
    return dict(name=name, N=int(N), M=int(src.size), **prof)


rng = np.random.default_rng(5)
meta = []
N, F = 1 << 20, 8
for win in (16, 256, 4096, 0):
    dst, *_ = make_graph(N, F, N if win == 0 else win, seed=5)
    src = np.repeat(np.arange(N, dtype=np.int64), F)
    meta.append(write_and_convert(f"syn_w{win}", N, src, dst.astype(np.int64), rng))
    print(meta[-1], flush=True)

# Real road network (SNAP roadNet-CA, undirected; edges listed in both directions)
url = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
gz = f"{DATA}/roadNet-CA.txt.gz"
# wget with timeout/retries; a plain urlretrieve hung for an hour on a stalled transfer
subprocess.run(["wget", "-q", "-c", "--timeout=60", "--tries=5", "-O", gz, url], check=True)
try:
    e = pd.read_csv(gz, sep="\t", comment="#", header=None, names=["s", "d"])
except (EOFError, OSError, gzip.BadGzipFile) as err:
    raise SystemExit(f"roadNet-CA download incomplete ({err}); synthetic graphs are still usable")
ids, inv = np.unique(np.concatenate([e.s.values, e.d.values]), return_inverse=True)
src, dst = inv[: len(e)].astype(np.int64), inv[len(e):].astype(np.int64)
meta.append(write_and_convert("roadNet-CA", ids.size, src, dst, rng))
print(meta[-1], flush=True)
json.dump(meta, open("../results/exp005_graphs.json", "w"), indent=1)
