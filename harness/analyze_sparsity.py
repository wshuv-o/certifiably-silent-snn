"""EXP-001 analysis: fit cost constants and compare measured vs predicted
sparsity crossovers s* (Phase 4, prediction P1).

Fits, per platform and locality window:
  SS:  T_SS(m) = a + k*m          per F;  across F: k = b + c_m*F
  DS:  T_DS    ~ constant in s    per F;  across F: T_DS = g + c_e*N*F
  chi = c_e / c_m  (dense per-edge cost / sparse per-message cost)
Crossover s*: measured = where median T_SS crosses T_DS (log-interpolated in s);
predicted = (T_DS - a) / (k*N) from the fitted SS line.
"""
import json, sys
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "../results/exp001"
df = pd.read_csv(path + ".csv")
med = df.groupby(["platform", "mode", "N", "F", "window", "s", "m"], as_index=False)["time_s"].median()
# Paired design: SS and DS run back-to-back in each rep, so the per-rep ratio
# cancels slow clock/thermal drift. Median of paired ratios decides the crossover.
keys = ["platform", "F", "window", "s", "rep"]
piv = df.pivot_table(index=keys, columns="mode", values="time_s").reset_index()
piv["ratio"] = piv["SS"] / piv["DS"]
pratio = piv.groupby(["platform", "F", "window", "s"])["ratio"].median()
iqr = df.groupby(["platform", "mode", "N", "F", "window", "s"])["time_s"].agg(
    lambda x: (np.percentile(x, 75) - np.percentile(x, 25)) / np.median(x))


from common import crossing


rows, consts = [], []
for (plat, win), g in med.groupby(["platform", "window"]):
    N = int(g.N.iloc[0])
    ks, ds_T, Fs = [], [], []
    for F, gf in g.groupby("F"):
        ss = gf[gf["mode"] == "SS"].sort_values("s")
        ds = gf[gf["mode"] == "DS"].sort_values("s")
        A = np.vstack([np.ones(len(ss)), ss.m.values]).T
        wts = 1.0 / ss.time_s.values               # relative-error weighting
        a, k = np.linalg.lstsq(A * wts[:, None], ss.time_s.values * wts, rcond=None)[0]
        T_ds = float(np.median(ds.time_s.values))
        ds_cv = float(np.std(ds.time_s.values) / T_ds)
        s_meas = crossing(ss.s.values, pratio.loc[(plat, F, win)].reindex(ss.s.values).values)
        s_pred = (T_ds - a) / (k * N)
        rows.append(dict(platform=plat, window=win, F=int(F), T_DS_ms=T_ds * 1e3, DS_cv_over_s=ds_cv,
                         SS_intercept_ms=a * 1e3, SS_per_active_ns=k * 1e9,
                         s_star_measured=s_meas, s_star_predicted=s_pred))
        ks.append(k); ds_T.append(T_ds); Fs.append(F)
    Fs = np.array(Fs, float)
    b, c_m = np.polyfit(Fs, ks, 1)[::-1]
    gamma, h = np.polyfit(Fs * N, ds_T, 1)[::-1]
    consts.append(dict(platform=plat, window=win, c_m_ns=c_m * 1e9, b_ns=b * 1e9,
                       c_e_ns=h * 1e9, dense_fixed_ms=gamma * 1e3, chi=h / c_m))

R = pd.DataFrame(rows)
C = pd.DataFrame(consts)
pd.set_option("display.width", 200)
print("\nPer-configuration crossover\n", R.round(4).to_string(index=False))
print("\nFitted cost constants (window 0 = random destinations)\n", C.round(4).to_string(index=False))
print("\nMax relative IQR across all cells:", round(float(iqr.max()), 3),
      " | median:", round(float(iqr.median()), 3))
with open(path + "_summary.json", "w") as f:
    json.dump({"crossovers": rows, "constants": consts,
               "iqr_rel_max": float(iqr.max()), "iqr_rel_median": float(iqr.median())}, f, indent=1)

