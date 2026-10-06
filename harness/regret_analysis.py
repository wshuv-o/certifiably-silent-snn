"""Phase 5 analysis on existing data (no new runs): how much do fixed regime rules
lose versus an oracle that always picks the faster mode?

Sparsity decision (EXP-004 v2 data, F=32, random dst, N=2^17..2^22, CPU+GPU):
  rules: always-dense, always-sparse, Ligra threshold (sparse iff frontier edges
  < |E|/20, i.e. s < 0.05 for a regular graph), and the best SINGLE threshold
  fitted per platform across all N (a tuned-but-static rule).
Asynchrony decision (EXP-005, Galois BFS): always-sync, always-async vs oracle.
Regret = T_rule / T_oracle, summarised as geometric mean and max over cells.
"""
import glob
import numpy as np
import pandas as pd

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/exp004_n[12][0-9].csv"))])
t = df.groupby(["platform", "N", "s", "mode"]).time_s.median().unstack("mode").reset_index()
t["oracle"] = t[["DS", "SS"]].min(axis=1)

def regret(choice_sparse):
    T = np.where(choice_sparse, t.SS, t.DS)
    return T / t.oracle

rules = {"always-dense": np.zeros(len(t), bool), "always-sparse": np.ones(len(t), bool),
         "Ligra s<0.05": (t.s < 0.05).values}
rows = []
for plat in ("cpu", "gpu"):
    m = (t.platform == plat).values
    best = None
    for thr in np.unique(t.s):                      # best single static threshold, in hindsight
        r = regret((t.s < thr).values)[m]
        g = np.exp(np.log(r).mean())
        if best is None or g < best[1]:
            best = (thr, g, r.max())
    for name, ch in rules.items():
        r = regret(ch)[m]
        rows.append(dict(platform=plat, rule=name, geomean_regret=np.exp(np.log(r).mean()), max_regret=r.max()))
    rows.append(dict(platform=plat, rule=f"best static thr (s<{best[0]:g}, hindsight)",
                     geomean_regret=best[1], max_regret=best[2]))
print(pd.DataFrame(rows).round(3).to_string(index=False))

a = pd.read_csv("../results/exp005_summary.csv")
a = a[a.kernel == "bfs"][["graph", "rounds", "T_sync_ms", "T_async_ms"]]
a["oracle"] = a[["T_sync_ms", "T_async_ms"]].min(axis=1)
a["regret_always_sync"] = a.T_sync_ms / a.oracle
a["regret_always_async"] = a.T_async_ms / a.oracle
print("\n", a.round(2).to_string(index=False))
