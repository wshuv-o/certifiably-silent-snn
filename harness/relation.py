#!/usr/bin/env python3
"""Single source of truth for the R_short / certified-silence relation.

Twice in one evening a plateau bound was typed into the manuscript and invalidated by the next run
that finished. The bounds are a function of the measured set, so they belong in one place that is
recomputed rather than in prose that is edited. Run this before quoting any of them.

Every point carries the run that produced it. Points are measured on the speaker-disjoint validation
set at K=4; constrained networks are fine-tuned from their own matched control.
"""
import json
import os
import sys

# (R_short, % of oracle certified, label, provenance)
POINTS = [
    (0.222, 95.8, "learnable d<=25, constrained",    "dx_c_wide_cert"),
    (0.243, 95.7, "learnable delays, constrained",   "dx_b_fix_cert (2 seeds)"),
    (0.260, 97.4, "local H=512, constrained",        "SCALE-LOCAL-001"),
    (0.261, 95.0, "D={2,4,8}, constrained",          "GOVERN-002"),
    (0.294, 96.3, "local H=1024, constrained",       "SCALE-LOCAL-001"),
    (0.297, 98.5, "local H=2048, constrained",       "SCALE-LOCAL-001"),
    (0.877, 90.0, "local H=2048, control",           "SCALE-LOCAL-001"),
    (1.006, 84.9, "local H=1024, control",           "SCALE-LOCAL-001"),
    (1.221, 34.6, "local H=512, control",            "SCALE-LOCAL-001"),
    (2.496,  1.5, "recipe minus dropout",            "ac_r_nodrop"),
    (2.773,  1.7, "recipe minus binning",            "ac_r_nobins"),
    (2.802, 15.5, "learnable delays d<=25, control", "dx_c_wide_ctrl"),
    (2.906,  0.9, "full reference recipe",           "ac_r_full"),
    (3.028,  2.4, "learnable delays, control",       "dx_b_fix_ctrl (2 seeds)"),
    (4.795,  0.5, "recipe minus batch norm",         "ac_r_nobn"),
    (4.813,  6.3, "D={2,4,8}, control",              "GOVERN-002"),
    (5.122,  6.1, "learned tau, control",            "ac_t_ctrl"),
    (5.631, 10.8, "arctangent surrogate only",       "ac_r_atan"),
    (7.905,  0.0, "unit delay D={1}",                "GOVERN-002"),
    (8.741,  3.8, "D={1,2,4,8}",                     "GOVERN-002"),
    (25.007, 0.2, "dense H=2048",                    "SCALE-LOCAL-001"),
]

LO, HI = 0.30, 2.40          # plateau edges, chosen once and held; the bounds below follow from them


def spearman(pts):
    n = len(pts)
    rx = sorted(range(n), key=lambda i: pts[i][0])
    ry = sorted(range(n), key=lambda i: pts[i][1])
    RX = [0] * n
    RY = [0] * n
    for r, i in enumerate(rx):
        RX[i] = r
    for r, i in enumerate(ry):
        RY[i] = r
    return 1 - 6 * sum((RX[i] - RY[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def main():
    pts = sorted(POINTS)
    lo = [p for p in pts if p[0] < LO]
    hi = [p for p in pts if p[0] > HI]
    mid = [p for p in pts if LO <= p[0] <= HI]
    out = dict(
        n=len(pts),
        rho=round(spearman(pts), 3),
        r_min=pts[0][0], r_max=pts[-1][0],
        fold=round(pts[-1][0] / pts[0][0]),
        lo_n=len(lo), lo_min=min(p[1] for p in lo), lo_max=max(p[1] for p in lo),
        hi_n=len(hi), hi_min=min(p[1] for p in hi), hi_max=max(p[1] for p in hi),
        mid=[(p[0], p[1]) for p in mid],
        mid_ordered=all(mid[i][1] >= mid[i + 1][1] for i in range(len(mid) - 1)),
    )
    print("n = %(n)d   Spearman rho = %(rho)s   R_short %(r_min)s to %(r_max)s (%(fold)d-fold)" % out)
    print("below %.2f (n=%d): %.1f-%.1f%% of oracle" % (LO, out["lo_n"], out["lo_min"], out["lo_max"]))
    print("above %.2f (n=%d): %.1f-%.1f%% of oracle" % (HI, out["hi_n"], out["hi_min"], out["hi_max"]))
    print("transition (n=%d): %s   monotone: %s"
          % (len(mid), ", ".join("%.3f->%.1f%%" % m for m in out["mid"]), out["mid_ordered"]))
    if "--json" in sys.argv:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "relation.json")
        json.dump(out, open(p, "w"), indent=1)
        print("wrote " + os.path.normpath(p))


if __name__ == "__main__":
    main()
