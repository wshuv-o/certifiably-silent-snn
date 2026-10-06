"""EXP-004 analysis: s* and operating-point cost ratio chi_op across N (working set)."""
import glob
import numpy as np
import pandas as pd
from common import crossing

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/exp004_n[12][0-9].csv"))])
F = int(df.F.iloc[0])
rows = []
for (plat, N), g in df.groupby(["platform", "N"]):
    piv = g.pivot_table(index=["s", "rep"], columns="mode", values="time_s").reset_index()
    piv["ratio"] = piv.SS / piv.DS
    pr = piv.groupby("s").ratio.median()
    s_star = crossing(pr.index.values, pr.values)
    ds_edge = g[g["mode"] == "DS"].time_s.median() / (N * F)
    ss = g[(g["mode"] == "SS") & (g.s >= 0.1)]
    ss_msg = np.median(ss.time_s / (ss.m * F))
    chi = ds_edge / ss_msg
    rows.append(dict(platform=plat, log2N=int(np.log2(N)), state_MB=4 * N / 2**20,
                     DS_ns_edge=ds_edge * 1e9, SS_ns_msg=ss_msg * 1e9, chi_op=chi,
                     s_star=s_star, s_over_chi=s_star / chi))
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
T.to_csv("../results/exp004_summary.csv", index=False)
