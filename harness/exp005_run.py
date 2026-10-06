"""EXP-005: asynchrony component on independent code (Galois).

For each graph: BFS (Sync vs Async) and SSSP (deltaStepBarrier vs deltaStep),
6 threads, active OpenMP-equivalent spinning is Galois-internal. Sync/async runs
are interleaved trial by trial (paired), with the order alternating.
G_async = median over trials of T_sync / T_async.
"""
import json, os, re, subprocess
import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/research/data/exp005")
GB = os.path.expanduser("~/research/refs/Galois/build/lonestar/analytics/cpu")
PAIRS = {"bfs": (f"{GB}/bfs/bfs-cpu", "Sync", "Async"),
         "sssp": (f"{GB}/sssp/sssp-cpu", "deltaStepBarrier", "deltaStep")}
TRIALS = 7


def galois_time_ms(out):
    """Whole-algorithm time: Galois 'STAT, (NULL), Timer_0, TMAX, <ms>' (the
    lonestar wrapper around the algorithm call; excludes graph loading and the
    sanity check). Per-region 'Time' stats are NOT comparable across modes: on
    syn_w256, Sync reported 'Sync, Time' = 14 ms while Timer_0 = 761 ms."""
    t = [float(l.split(",")[-1]) for l in out.splitlines()
         if l.startswith("STAT, (NULL), Timer_0,")]
    if len(t) != 1:
        raise RuntimeError(f"expected exactly one Timer_0 line, got {t}")
    return t[0]


def run(binary, algo, gr):
    r = subprocess.run([binary, f"--algo={algo}", "-t", "6", "--startNode=0", gr],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return galois_time_ms(r.stdout), r.stdout


graphs = json.load(open("../results/exp005_graphs.json"))
rows = []
for g in graphs:
    gr = f"{DATA}/{g['name']}.gr"
    for kern, (binary, sync, asyn) in PAIRS.items():
        for t in range(TRIALS + 1):          # trial 0 is warm-up (page cache)
            for algo in ((sync, asyn) if t % 2 == 0 else (asyn, sync)):
                ms, out = run(binary, algo, gr)
                if t == 0 and algo == sync and g is graphs[0] and kern == "bfs":
                    print("\n".join(l for l in out.splitlines() if l.startswith("STAT"))[:1500])
                if t > 0:
                    rows.append((g["name"], kern, algo, "sync" if algo == sync else "async", t, ms))
        print(g["name"], kern, "done", flush=True)

df = pd.DataFrame(rows, columns=["graph", "kernel", "algo", "mode", "trial", "time_ms"])
df.to_csv("../results/exp005.csv", index=False)
piv = df.pivot_table(index=["graph", "kernel", "trial"], columns="mode", values="time_ms").reset_index()
piv["G_async"] = piv["sync"] / piv["async"]
S = piv.groupby(["graph", "kernel"]).agg(T_sync_ms=("sync", "median"), T_async_ms=("async", "median"),
                                         G_async=("G_async", "median"),
                                         G_q25=("G_async", lambda x: x.quantile(.25)),
                                         G_q75=("G_async", lambda x: x.quantile(.75))).reset_index()
S = S.merge(pd.DataFrame(graphs).rename(columns={"name": "graph"}), on="graph")
print(S.round(4).to_string(index=False))
S.to_csv("../results/exp005_summary.csv", index=False)
