"""Strawman guard (Phase 6 acceptance criterion): compare our CPU kernels with
gapbs on the SAME graph, using per-edge costs.

  dense : our DS pull sweep   vs  gapbs PageRank (pull sweep), ns per edge per iteration
  sparse: our SS push         vs  gapbs BFS top-down steps (frontier push), ns per edge

Writes the graph as an edge list, converts with gapbs `converter`, runs gapbs with
the same thread protocol (6 threads, active OpenMP wait), and parses timings.
"""
import os, re, subprocess, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
os.environ.setdefault("OMP_WAIT_POLICY", "active")
import numpy as np
import pandas as pd
from graphgen import make_graph, sample_active
from cpu_kernels import CPUEngine

N, F, SEED = 1 << 20, 32, 0
GAP = os.path.expanduser("~/research/refs/gapbs")
DATA = os.path.expanduser("~/research/data")
os.makedirs(DATA, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="6", OMP_WAIT_POLICY="active")

dst, w, in_ptr, in_idx, in_w = make_graph(N, F, N, seed=SEED)
sg = f"{DATA}/rand_n20_f{F}.sg"
if not os.path.exists(sg):
    el = f"{DATA}/rand_n20_f{F}.el"
    src = np.repeat(np.arange(N, dtype=np.int64), F)
    pd.DataFrame({"s": src, "d": dst}).to_csv(el, sep=" ", header=False, index=False)
    subprocess.run([f"{GAP}/converter", "-f", el, "-b", sg], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    os.remove(el)

def run(cmd):
    return subprocess.run(cmd, check=True, env=env, capture_output=True, text=True).stdout

# ---- gapbs PageRank: fixed 20 iterations (tolerance 0), 5 trials
out = run([f"{GAP}/pr", "-f", sg, "-n", "5", "-i", "20", "-t", "0"])
M = int(re.search(r"(\d+) directed edges", out).group(1))
trials = [float(x) for x in re.findall(r"Trial Time:\s+([\d.]+)", out)]
pr_ns_edge = np.median(trials) / 20 / M * 1e9

# ---- gapbs BFS with step logging: keep top-down steps with large frontiers
out = run([f"{GAP}/bfs", "-f", sg, "-n", "8", "-l"])
# gapbs logs the OUTPUT frontier of each td step (queue.size() after slide_window,
# bfs.cc:171), so a step's input frontier is the previous step's logged count.
# A td step that follows a bottom-up phase has an unlogged input size and is skipped.
td, prev = [], None
for line in out.splitlines():
    p = line.split()
    if not p:
        continue
    if p[0] == "Source":
        prev = 1
    elif p[0] == "td" and len(p) == 3:
        if prev is not None:
            td.append((prev, float(p[2])))
        prev = int(p[1])
    elif p[0] in ("bu", "e", "c"):
        prev = None
td = np.array(td)
big = td[td[:, 0] >= 1000] if len(td) else td   # input frontier size in vertices
bfs_ns_edge = np.median(big[:, 1] / (big[:, 0] * (M / N))) * 1e9 if len(big) else float("nan")

# ---- ours on the same graph
eng = CPUEngine(N, F, dst, w, in_ptr, in_idx, in_w)
state0 = np.random.default_rng(1).uniform(0, 1, N).astype(np.float32)
def t_step(fn, act, reps=15):
    st = state0.copy(); ts = []
    for r in range(reps + 2):
        np.copyto(st, state0); t0 = time.perf_counter(); fn(act, st, 1.0); ts.append(time.perf_counter() - t0)
    return np.median(ts[2:])
rng = np.random.default_rng(2)
ds_ns_edge = t_step(eng.ds, sample_active(N, 1.0, rng)) / (N * F) * 1e9
ss = {s: t_step(eng.ss, sample_active(N, s, rng)) for s in (0.01, 0.03, 0.1)}
ss_ns_edge = {s: t / (round(s * N) * F) * 1e9 for s, t in ss.items()}

from cpu_kernels import CPPEngine
from graphgen import reference_step
cpp = CPPEngine(N, F, dst, w, in_ptr, in_idx, in_w)
act = sample_active(N, 0.03, np.random.default_rng(5))
ref_state, ref_next = reference_step(act, dst, w, F, state0, 1.0)
for mode in ("ds", "ss"):
    st = state0.copy(); nx = getattr(cpp, mode)(act, st, 1.0)
    ok = np.allclose(st, ref_state, rtol=1e-4, atol=1e-4) and len(set(nx.tolist()) ^ set(ref_next.tolist())) <= 100
    print(f"C++ {mode.upper()} verification vs NumPy reference: {'PASS' if ok else 'FAIL'}")
cpp_ds = t_step(cpp.ds, sample_active(N, 1.0, rng)) / (N * F) * 1e9
cpp_ss = {s: t_step(cpp.ss, sample_active(N, s, rng)) / (round(s * N) * F) * 1e9 for s in (0.01, 0.03, 0.1)}
print(f"C++ DS ns/edge {cpp_ds:.3f} (ratio to gapbs PR {cpp_ds/pr_ns_edge:.2f})")
for s, v in cpp_ss.items():
    print(f"C++ SS s={s:<5} ns/edge {v:.3f} (ratio to gapbs BFS-td {v/bfs_ns_edge:.2f})")

print(f"graph: N={N} F={F} gapbs edges M={M} (ours {N*F})")
print(f"DENSE  ns/edge: ours DS {ds_ns_edge:.3f} | gapbs PR {pr_ns_edge:.3f} | ratio ours/gapbs {ds_ns_edge/pr_ns_edge:.2f}")
print(f"SPARSE ns/edge: gapbs BFS-td (input frontier>=1e3, {len(big)} steps) {bfs_ns_edge:.3f}")
for s, v in ss_ns_edge.items():
    print(f"               ours SS s={s:<5} {v:.3f} | ratio ours/gapbs {v/bfs_ns_edge:.2f}")
print("td steps (frontier, sec):", np.round(td, 5).tolist()[:12])
