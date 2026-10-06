"""EXP-006: compose pre-registered predictions from primitives; compare with EXP-004."""
import json
import numpy as np
import pandas as pd

F = 32
mic = json.load(open("../results/exp006_micro.json"))
meas = pd.read_csv("../results/exp004_summary.csv")
rows = []
for k in [17, 18, 19, 20, 21, 22]:
    N = 1 << k
    g = mic["gpu"][str(k)]
    L = mic["gpu"]["launch_us"] * 1e-6
    ce, cm = g["gather_ns"] * 1e-9, g["atomic_ns"] * 1e-9
    s_gpu = (5 * L + N * F * ce - 3 * L) / (N * F * cm)
    c = mic["cpu"][str(k)]
    s_cpu = c["gather_ns"] / (2 * mic["cpu"]["stream_ns"] + c["rmw_ns"])
    for plat, sp in (("gpu", s_gpu), ("cpu", s_cpu)):
        m = float(meas[(meas.platform == plat) & (meas.log2N == k)].s_star.iloc[0])
        if np.isinf(m):
            err = 0.0 if sp > 1 else np.nan
        else:
            err = (sp - m) / m
        rows.append(dict(platform=plat, log2N=k, s_pred=sp, s_meas=m, rel_err=err))
T = pd.DataFrame(rows).sort_values(["platform", "log2N"])
print(T.round(4).to_string(index=False))
e = T.rel_err.abs()
print("criterion 1: median |rel err| =", round(float(e.median()), 3), "(<= 0.30)",
      "| unmatched-inf cases:", int(T.rel_err.isna().sum()))
gp = T[T.platform == "gpu"].set_index("log2N")
print("criterion 2: predicted GPU argmin =", int(gp.s_pred.idxmin()), "(measured 19, +/-1)")
cp_ = T[T.platform == "cpu"].set_index("log2N")
print("criterion 3: predicted CPU rise 2^20->2^22 =",
      round(float(cp_.s_pred[22] / cp_.s_pred[20] - 1), 3), "(>= 0.30)")
T.to_csv("../results/exp006_comparison.csv", index=False)
