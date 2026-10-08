#!/usr/bin/env python3
"""Regenerate every manuscript number derived from the R_short relation, from the data.

Twice tonight a bound was typed into the prose and invalidated by the next run to finish, and a third
time a table row carried a superseded measurement. Those were transcription errors, not measurement
errors, and the fix is to stop transcribing. This reads the recorded results, recomputes table 1's
learnable-delay rows and every derived bound, and rewrites them in place.

It is idempotent and it asserts before it writes: if a number in the manuscript is not where this
script expects it, nothing is written and it says which one.

Usage:  python finalise_relation.py [--dry-run]
"""
import glob
import io
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "research", "paper", "manuscript_nce.tex")
RESULTS = os.path.join(HERE, "..", "results")
DRY = "--dry-run" in sys.argv


def seeds(prefix):
    """All seed runs of one configuration, as recorded by the harness."""
    out = []
    for f in glob.glob(os.path.join(RESULTS, "dc_%s*.json" % prefix)):
        d = json.load(open(f))
        if d.get("violations", 0) != 0:
            raise SystemExit("REFUSING: %s reports %d soundness violations" % (f, d["violations"]))
        out.append(d)
    return sorted(out, key=lambda d: d["seed"])


def row(ds):
    """Table 1 row fields, averaged over seeds."""
    m = lambda k: st.mean(d[k] for d in ds)
    return dict(
        n=len(ds),
        acc=100 * m("acc_val"),
        cert=100 * m("core_cert"),
        oracle=100 * m("core_oracle"),
        pct=st.mean(100 * d["core_cert"] / (d["core_oracle"] or 1e-9) for d in ds),
        rshort=m("R_short"),
        rtot=m("R_mean"),
    )


def main():
    ctrl, cert = seeds("b_fix_ctrl"), seeds("b_fix_cert")
    if not ctrl or not cert:
        raise SystemExit("no learnable-delay results found in %s" % RESULTS)
    if len(ctrl) != len(cert):
        raise SystemExit("seed counts differ: %d control, %d constrained" % (len(ctrl), len(cert)))
    c, q = row(ctrl), row(cert)
    print("learnable-delay rows over %d seeds" % c["n"])
    for name, r in (("control", c), ("constrained", q)):
        print("  %-12s acc %.2f  cert %.2f%%  oracle %.2f%%  %%or %.1f  R_short %.3f  Rtot %.2f"
              % (name, r["acc"], r["cert"], r["oracle"], r["pct"], r["rshort"], r["rtot"]))

    # the relation, recomputed with those seed means substituted in
    sys.path.insert(0, HERE)
    import relation
    pts = []
    for p in relation.POINTS:
        if "dx_b_fix_ctrl" in p[3]:
            pts.append((round(c["rshort"], 3), round(c["pct"], 1), p[2], p[3]))
        elif "dx_b_fix_cert" in p[3]:
            pts.append((round(q["rshort"], 3), round(q["pct"], 1), p[2], p[3]))
        else:
            pts.append(p)
    pts.sort()
    rho = relation.spearman(pts)
    lo = [p for p in pts if p[0] < relation.LO]
    hi = [p for p in pts if p[0] > relation.HI]
    mid = [p for p in pts if relation.LO <= p[0] <= relation.HI]
    print("\nrelation: n=%d  rho=%.3f  below %.2f (n=%d) %.1f-%.1f%%  above %.2f (n=%d) %.1f-%.1f%%"
          % (len(pts), rho, relation.LO, len(lo), min(p[1] for p in lo), max(p[1] for p in lo),
             relation.HI, len(hi), min(p[1] for p in hi), max(p[1] for p in hi)))
    print("transition: " + ", ".join("%.3f->%.1f%%" % (p[0], p[1]) for p in mid)
          + ("  (monotone)" if all(mid[i][1] >= mid[i + 1][1] for i in range(len(mid) - 1)) else "  (NOT monotone)"))

    s = io.open(TEX, encoding="utf-8").read()
    edits = []

    def sub(pattern, repl, label):
        nonlocal s
        n = len(re.findall(pattern, s))
        if n != 1:
            raise SystemExit("REFUSING: %s matched %d times, expected 1" % (label, n))
        s = re.sub(pattern, repl.replace("\\", "\\\\"), s, count=1)
        edits.append(label)

    sub(r"learnable delays, control             & [\d.]+ & [\d.]+\\% +& [\d.]+\\% & [\d.]+\\% +& [\d.]+ & [\d.]+ \\\\",
        "learnable delays, control             & %.2f & %.2f\\%%  & %.2f\\%% & %.1f\\%%  & %.3f & %.2f \\\\"
        % (c["acc"], c["cert"], c["oracle"], c["pct"], c["rshort"], c["rtot"]), "table 1 control")

    sub(r"learnable delays, constrained         & [\d.]+ & [\d.]+\\% & [\d.]+\\% & \\textbf\{[\d.]+\\%\} & \\textbf\{[\d.]+\} & [\d.]+ \\\\",
        "learnable delays, constrained         & %.2f & %.2f\\%% & %.2f\\%% & \\textbf{%.1f\\%%} & \\textbf{%.3f} & %.2f \\\\"
        % (q["acc"], q["cert"], q["oracle"], q["pct"], q["rshort"], q["rtot"]), "table 1 constrained")

    sub(r"mean over three seeds, except the learnable-delay rows which are seed 1 pending replication\.",
        "mean over three seeds.", "table 1 caption")

    sub(r"Across \w+ networks spanning a hundredfold range of \$R_\{\\mathrm\{short\}\}\$, certified silence declines monotonically with it \(Spearman \$\\rho=-[\d.]+\$\)",
        "Across %d networks spanning a hundredfold range of $R_{\\mathrm{short}}$, certified silence declines monotonically with it (Spearman $\\rho=%.3f$)" % (len(pts), rho),
        "abstract rho")

    sub(r"\w+ networks spanning 0\.2\d+ to 25\.0 \(Spearman \$\\rho=-[\d.]+\$\)",
        "%d networks spanning %.3f to 25.0 (Spearman $\\rho=%.3f$)" % (len(pts), pts[0][0], rho),
        "contributions rho")

    sub(r"Pooling all of them gives \$\\rho=-[\d.]+\$ over \w+ networks",
        "Pooling all of them gives $\\rho=%.3f$ over %d networks" % (rho, len(pts)), "results rho")

    if DRY:
        print("\ndry run; %d edits would be applied: %s" % (len(edits), ", ".join(edits)))
        return
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("\napplied %d edits: %s" % (len(edits), ", ".join(edits)))


if __name__ == "__main__":
    main()
