#!/usr/bin/env python3
"""Robustness of the 23-network relation to network families (supplement, section 1).

Groups the points of relation.py into the eight families named in the supplement, recomputes the
Spearman correlation with each family left out, and with one median network per family.
Usage: python relation_families.py [--json]"""
import json
import os
import sys
import numpy as np
from relation import POINTS, spearman

FAMILY = {
    "learnable": lambda l: l.startswith("learnable"),
    "locally connected": lambda l: l.startswith("local"),
    "D={2,4,8}": lambda l: l.startswith("D={2,4,8}"),
    "learnable time constants": lambda l: l.startswith("learned tau"),
    "recipe variants": lambda l: l.startswith("recipe") or l.startswith("full reference") or l.startswith("arctangent"),
    "unit delay": lambda l: l.startswith("unit delay"),
    "D={1,2,4,8}": lambda l: l.startswith("D={1,2,4,8}"),
    "dense H=2048": lambda l: l.startswith("dense"),
}


def family(label):
    hits = [f for f, test in FAMILY.items() if test(label)]
    assert len(hits) == 1, label
    return hits[0]


def main():
    groups = {}
    for p in POINTS:
        groups.setdefault(family(p[2]), []).append(p)
    assert sum(len(g) for g in groups.values()) == len(POINTS) == 23
    loo = {f: round(spearman([p for p in POINTS if family(p[2]) != f]), 3) for f in groups}
    medians = []
    for f, g in groups.items():
        g = sorted(g)
        medians.append(g[len(g) // 2] if len(g) % 2 else (float(np.mean([g[len(g) // 2 - 1][0], g[len(g) // 2][0]])),
                                                          float(np.mean([g[len(g) // 2 - 1][1], g[len(g) // 2][1]]))))
    out = dict(rho_all=round(spearman(POINTS), 3), leave_one_family_out=loo,
               loo_min=min(loo.values()), loo_max=max(loo.values()),
               median_per_family=round(spearman(medians), 3), families={f: len(g) for f, g in groups.items()})
    for f, r in loo.items():
        print(f"without {f:26s} (n={len(groups[f])}): rho = {r}")
    print("range", out["loo_max"], "to", out["loo_min"], "; one median network per family: rho =", out["median_per_family"])
    if "--json" in sys.argv:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "relation_families.json")
        json.dump(out, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
