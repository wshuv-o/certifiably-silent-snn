"""EXP-002: out-of-sample test of the fitted cost model (prediction P1).

  predict : write s* predictions for unseen (F, N, seeds), using only constants
            fitted on EXP-001 v2 (N = 2^20, F in {8, 32, 64}, seed 0)
  compare : after the runs, compare predicted vs measured s*

Model (Phase 4 section 5, kappa ~ 1):
  T_DS(N, F) = g * (N / N0) + c_e * N * F     (fixed term assumed proportional to N)
  T_SS(m, F) = a + (b + c_m * F) * m          (a = mean fitted SS intercept)
  s*        = (T_DS - a) / ((b + c_m * F) * N)
"""
import json, sys
import numpy as np
import pandas as pd

R = "../results/"
N0 = 1 << 20
CASES = [(1 << 20, 16), (1 << 20, 48), (1 << 19, 32)]


def predict():
    summ = json.load(open(R + "exp001v2_summary.json"))
    a_mean = pd.DataFrame(summ["crossovers"]).groupby(["platform", "window"]).SS_intercept_ms.mean() / 1e3
    out = []
    for c in summ["constants"]:
        plat, win = c["platform"], c["window"]
        b, cm, ce, g = c["b_ns"] * 1e-9, c["c_m_ns"] * 1e-9, c["c_e_ns"] * 1e-9, c["dense_fixed_ms"] * 1e-3
        a = a_mean[(plat, win)]
        for N, F in CASES:
            T_ds = g * N / N0 + ce * N * F
            s = (T_ds - a) / ((b + cm * F) * N)
            out.append(dict(platform=plat, window=win, N=N, F=F, s_star_pred=s))
    json.dump(out, open(R + "exp002_predictions.json", "w"), indent=1)
    print(pd.DataFrame(out).round(4).to_string(index=False))


def compare():
    from common import crossing
    pred = pd.DataFrame(json.load(open(R + "exp002_predictions.json")))
    df = pd.concat([pd.read_csv(R + f) for f in ("exp002a.csv", "exp002b.csv")])
    keys = ["platform", "N", "F", "window", "s", "rep"]
    piv = df.pivot_table(index=keys, columns="mode", values="time_s").reset_index()
    piv["ratio"] = piv.SS / piv.DS
    pr = piv.groupby(["platform", "N", "F", "window", "s"]).ratio.median().reset_index()
    rows = []
    for (plat, N, F, win), g in pr.groupby(["platform", "N", "F", "window"]):
        g = g.sort_values("s")
        meas = crossing(g.s.values, g.ratio.values)
        p = pred[(pred.platform == plat) & (pred.window == win) & (pred.N == N) & (pred.F == F)].s_star_pred
        p = float(p.iloc[0]) if len(p) else np.nan
        rows.append(dict(platform=plat, window=win, N=N, F=F, s_pred=p, s_meas=meas,
                         rel_err=(meas - p) / p))
    T = pd.DataFrame(rows)
    print(T.round(4).to_string(index=False))
    print("median |rel err|:", round(float(T.rel_err.abs().median()), 3),
          " max:", round(float(T.rel_err.abs().max()), 3))
    T.to_csv(R + "exp002_comparison.csv", index=False)


if __name__ == "__main__":
    {"predict": predict, "compare": compare}[sys.argv[1]]()

