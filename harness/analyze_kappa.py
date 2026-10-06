"""EXP-003 analysis: sparsity crossover s* as a function of spatial imbalance kappa."""
import sys
import numpy as np
import pandas as pd
from common import crossing

path = sys.argv[1] if len(sys.argv) > 1 else "../results/exp003"
df = pd.read_csv(path + ".csv")
keys = ["platform", "window", "spread", "s", "rep"]
piv = df.pivot_table(index=keys, columns="mode", values="time_s").reset_index()
piv["ratio"] = piv.SS / piv.DS
pr = piv.groupby(["platform", "window", "spread", "s"]).ratio.median().reset_index()
kap = df.groupby(["platform", "window", "spread", "s"]).kappa.first()

rows = []
for (plat, win, sp), g in pr.groupby(["platform", "window", "spread"]):
    g = g.sort_values("s")
    s_star = crossing(g.s.values, g.ratio.values)
    # kappa at the grid point nearest the crossing (kappa depends on s)
    s_near = g.s.values[np.argmin(np.abs(np.log(g.s.values) - np.log(max(s_star, 1e-4))))] \
        if np.isfinite(s_star) and s_star > 0 else np.nan
    k = float(kap.loc[(plat, win, sp, s_near)]) if np.isfinite(s_near) else np.nan
    rows.append(dict(platform=plat, window=win, spread=sp, kappa_at_cross=k, s_star=s_star))
T = pd.DataFrame(rows)
base = T[T.spread == 1.0].set_index(["platform", "window"]).s_star
T["s_star_rel"] = [r.s_star / base[(r.platform, r.window)] for r in T.itertuples()]
T["inv_kappa"] = 1.0 / T.kappa_at_cross
print(T.round(3).to_string(index=False))
T.to_csv(path + "_summary.csv", index=False)

