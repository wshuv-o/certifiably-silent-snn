"""SWEEP-001: tabulate validation results and apply the pre-registered selection rule.
Rule: among configs with val accuracy within 1.0 point of the val control (ref_ctrl), pick the
highest val certified core fraction (cores that others wait on); ties -> higher val accuracy.
"""
import glob, json
import pandas as pd

rows = []
for f in sorted(glob.glob("../results/sweep_*_s1.json")):
    r = json.load(open(f))
    if r["tag"] == "smoke":
        continue
    rows.append(dict(tag=r["tag"], acc_val=100 * r["acc_val"], cert=100 * r["core_cert"],
                     oracle=100 * r["core_oracle"], allcore=100 * r.get("allcore_cert", r["core_cert"]),
                     viol=r["violations"], R=r["R_mean"], rate=100 * r["rate"], norec=100 * r["acc_norec"]))
T = pd.DataFrame(rows).sort_values("tag")
pd.set_option("display.width", 200)
print(T.round(2).to_string(index=False))
if "ref_ctrl" in set(T.tag):
    ctrl = float(T[T.tag == "ref_ctrl"].acc_val.iloc[0])
    ok = T[(T.acc_val >= ctrl - 1.0) & (T.tag != "ref_ctrl")].sort_values(["cert", "acc_val"], ascending=False)
    print(f"\nval control = {ctrl:.2f}; eligible (within 1.0 pt): {list(ok.tag)}")
    if len(ok):
        print("SELECTED (so far):", ok.iloc[0].tag, "cert", round(ok.iloc[0].cert, 2), "acc_val", round(ok.iloc[0].acc_val, 2))
T.to_csv("../results/sweep_table.csv", index=False)
