"""Rebuild the governing-relation table from the GOVERN-002 runs.

Every architecture is now trained at the common budget (150 epochs, AUG=2) on seeds 1-3. Seed 1 of
rows 2-6 already existed at that budget and is reused; row 1 had no matched run at any seed.

Emits the LaTeX table body and the per-architecture means the figure uses, so the table and the
figure cannot drift apart.

    python harness/rebuild_table1.py
"""
import glob
import json
import os
import statistics as st

LOGS = os.path.expanduser("~/research/logs")
if not os.path.isdir(LOGS):                      # running from Windows against the WSL filesystem
    LOGS = r"\\wsl$\Ubuntu\home\esme_abha\research\logs"

# label, latex name, [log basenames in seed order]
ARCHS = [
    ("unit",   r"unit delay $\mathcal{D}=\{1\}$",
     ["g2_u1_s1", "g2_u1_s2", "g2_u1_s3"]),
    ("m4",     r"$\mathcal{D}=\{1,2,4,8\}$",
     ["d1_D2_d1248_long", "g2_m4_s2", "g2_m4_s3"]),
    ("t3ctrl", r"$\mathcal{D}=\{2,4,8\}$, control",
     ["d2_E3_mindelay2_long", "g2_t3c_s2", "g2_t3c_s3"]),
    ("t3cons", r"$\mathcal{D}=\{2,4,8\}$, constrained",
     ["d3_F2_cert10", "g2_t3k_s2", "g2_t3k_s3"]),
    ("dcctrl", r"learnable delays, control",
     ["dc_DC1_ctrl", "g2_dcc_s2", "g2_dcc_s3"]),
    ("dccons", r"learnable delays, constrained",
     ["dc_DC1_cert", "g2_dck_s2", "g2_dck_s3"]),
]


def load(basename):
    path = os.path.join(LOGS, basename + ".log")
    if not os.path.exists(path):
        return None
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("RESULT"):
                return json.loads(line[7:])
    return None


def fmt(vals, digits=2):
    """mean, with the range appended when more than one seed contributed."""
    m = st.mean(vals)
    if len(vals) == 1:
        return ("%." + str(digits) + "f") % m
    return ("%." + str(digits) + "f") % m


rows, missing = [], []
for key, name, basenames in ARCHS:
    runs = [(b, load(b)) for b in basenames]
    got = [(b, r) for b, r in runs if r is not None]
    for b, r in runs:
        if r is None:
            missing.append(b)
    if not got:
        continue
    acc = [r["acc_val"] * 100 for _, r in got]
    cert = [r["core_cert"] * 100 for _, r in got]
    orac = [r["core_oracle"] * 100 for _, r in got]
    frac = [100.0 * r["core_cert"] / r["core_oracle"] for _, r in got]
    rsh = [r["R_short"] for _, r in got]
    tot = [sum(r["R_per_delay"]) for _, r in got]
    viol = sum(r["violations"] for _, r in got)
    rows.append(dict(key=key, name=name, n=len(got), acc=acc, cert=cert, oracle=orac,
                     frac=frac, rshort=rsh, total=tot, viol=viol))

print("=" * 100)
print("%-34s %2s %7s %8s %8s %8s %8s %8s %5s" %
      ("architecture", "n", "acc", "cert", "oracle", "%oracle", "R_short", "total R", "viol"))
print("-" * 100)
for r in rows:
    print("%-34s %2d %7.2f %7.2f%% %7.2f%% %7.1f%% %8.3f %8.2f %5d" %
          (r["key"], r["n"], st.mean(r["acc"]), st.mean(r["cert"]), st.mean(r["oracle"]),
           st.mean(r["frac"]), st.mean(r["rshort"]), st.mean(r["total"]), r["viol"]))
    if r["n"] > 1:
        print("%-34s    range  %.2f-%.2f   cert %.2f-%.2f   R_short %.3f-%.3f" %
              ("", min(r["acc"]), max(r["acc"]), min(r["cert"]), max(r["cert"]),
               min(r["rshort"]), max(r["rshort"])))
print("=" * 100)

if missing:
    print("\nSTILL MISSING (" + str(len(missing)) + "): " + ", ".join(missing))

print("\n--- LaTeX table body ---")
for r in rows:
    print("%s & %s & %s\\%% & %s\\%% & %s\\%% & %s & %s \\\\" % (
        r["name"], fmt(r["acc"]), fmt(r["cert"]), fmt(r["oracle"]),
        fmt(r["frac"], 1), fmt(r["rshort"], 3), fmt(r["total"], 2)))

print("\n--- figure 2 panel (a,b) rows: (index, %oracle, R_short, total R, has penalty) ---")
for i, r in enumerate(rows, start=1):
    pen = "True " if "cons" in r["key"] else "False"
    print("        (%d, %5.1f, %6.3f, %5.2f, %s),   # %s  n=%d" % (
        i, st.mean(r["frac"]), st.mean(r["rshort"]), st.mean(r["total"]), pen, r["key"], r["n"]))

bar1 = [r for r in rows if r["key"] == "unit"]
if bar1:
    worst = max(bar1[0]["frac"])
    print("\nPRE-REGISTERED BAR 1 (unit delay certifies < 10%% of oracle on every seed): "
          "worst seed = %.2f%% -> %s" % (worst, "PASS" if worst < 10 else "FAIL"))
    print("  unit-delay accuracy across seeds: " +
          ", ".join("%.2f" % a for a in bar1[0]["acc"]))
