"""Figures for the N3 manuscript. Reads result JSONs (and logged values where noted).
Palette validated with the dataviz validator (light): blue, orange, aqua, yellow, magenta.
Low-contrast hues (aqua/yellow/magenta) always carry direct value labels (relief rule).
"""
import os, json, glob
os.environ["LOCAL"] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

R = "../results/"; OUT = "../research/paper/figures/"; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
J = lambda f: json.load(open(R + f))


def label_bars(ax, bars, fmt="{:.0f}", dy=1.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7, color=INK)


# ---------------- Fig 2: the gap (untrained networks) ----------------
def fig_gap():
    d3, d3b = J("pilot003.json"), J("pilot003b.json")
    Ks = ["2", "4", "8"]; x = np.arange(len(Ks)); w = 0.2
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    series = [("Dense: actually silent", [100 * d3["results"][k]["oracle"]["partition32"] for k in Ks], ORANGE, None),
              ("Dense: provably silent", [100 * d3["results"][k]["recursive"]["partition32"] for k in Ks], ORANGE, "////"),
              ("Local: actually silent", [100 * d3b["results"][k]["oracle"]["partition32"] for k in Ks], BLUE, None),
              ("Local: provably silent", [100 * d3b["results"][k]["recursive"]["partition32"] for k in Ks], BLUE, "////")]
    for i, (lab, vals, col, hatch) in enumerate(series):
        b = ax.bar(x + (i - 1.5) * w, vals, w * 0.9, label=lab, color=col if hatch is None else "white",
                   edgecolor=col, hatch=hatch, linewidth=1.0)
        label_bars(ax, b, "{:.0f}", 0.8)
    ax.set_xticks(x, [f"K = {k}" for k in Ks]); ax.set_ylabel("Core-steps silent for K steps (%)"); ax.set_ylim(0, 85)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Untrained RSNNs: silent, but not provably", fontsize=8.5, color=INK, loc="left")
    fig.savefig(OUT + "fig2_gap.png"); plt.close(fig)


# ---------------- Fig 3: methods (small model, 3 seeds) ----------------
SEED1_LOG = {"control": dict(acc=0.682, cert=0.000, orac=0.611),   # PILOT-005 log (seed-1 file overwritten)
             "cert": dict(acc=0.694, cert=0.579, orac=0.613)}


def small_methods():
    out = {}
    def nb(d): r = d["results"]["4"]; return d["accuracy"], r["recursive"]["neighbours_silent"], r["oracle"]["neighbours_silent"]
    for name, pat in (("control", "pilot004_l0_s{}.json"), ("cert", "pilot004_l0.1_s{}.json"),
                      ("clamp", "alt001_clamp_s{}.json"), ("l1", "alt001_l1_s{}.json"), ("rate", "alt001_rate_s{}.json")):
        rows = []
        for s in (1, 2, 3):
            if s == 1 and name in SEED1_LOG:
                v = SEED1_LOG[name]; rows.append((v["acc"], v["cert"], v["orac"]))
            else:
                rows.append(nb(J(pat.format(s))))
        out[name] = np.array(rows)
    return out


def fig_methods():
    m = small_methods()
    names = ["control", "rate", "cert", "clamp", "l1"]
    labels = ["No\nconstraint", "Strong rate\npenalty", "Certificate\nloss", "Excitation\ncap", "L1 on\nexcitation"]
    cols = [INK2, MAGENTA, BLUE, AQUA, YELLOW]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]; x = np.arange(len(names))
    cert = [100 * m[n][:, 1].mean() for n in names]; cerr = [100 * m[n][:, 1].std() for n in names]
    orac = [100 * m[n][:, 2].mean() for n in names]
    b = ax.bar(x, cert, 0.6, color=cols, yerr=cerr, capsize=2, error_kw=dict(lw=0.8, ecolor=INK2))
    ax.scatter(x, orac, marker="_", s=260, color=INK, linewidths=1.6, zorder=3, label="Actually silent (oracle)")
    for xi, v, col in zip(x, cert, cols):
        inside = v > 10
        ax.text(xi, v - 7 if inside else v + 2.0, f"{v:.1f}", ha="center", fontsize=7,
                color=("white" if col in (BLUE, INK2) else INK) if inside else INK)
    ax.set_xticks(x, labels, fontsize=7); ax.set_ylabel("Provably silent neighbour-steps, K = 4 (%)")
    ax.set_ylim(0, 75); ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("a  Certifiability (3 seeds, mean ± sd)", fontsize=8.5, loc="left", color=INK)
    ax = axs[1]
    acc = [100 * m[n][:, 0].mean() for n in names]; aerr = [100 * m[n][:, 0].std() for n in names]
    b = ax.bar(x, acc, 0.6, color=cols, yerr=aerr, capsize=2, error_kw=dict(lw=0.8, ecolor=INK2))
    for xi, v, col in zip(x, acc, cols):
        ax.text(xi, v - 3.0, f"{v:.1f}", ha="center", fontsize=7, color="white" if col in (BLUE, INK2) else INK)
    ax.set_xticks(x, labels, fontsize=7); ax.set_ylabel("SHD test accuracy (%)"); ax.set_ylim(50, 76)
    ax.set_title("b  Accuracy", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(OUT + "fig3_methods.png"); plt.close(fig)
    return m


# ---------------- Fig 4: trade-off (strong model, seed 1) ----------------
def fig_tradeoff():
    pts = [("No constraint", "s2_l0_s1.json"), ("λ = 0.03", "s2_l0.03_s1.json"), ("λ = 0.1", "s2_l0.1_s1.json"),
           ("λ = 0.3", "s2_l0.3_s1.json"), ("λ = 1", "s2_l1_s1.json"), ("L1", "s2_shd_l1_l0_s1.json"),
           ("ExCap", "s2_shd_clamp_l0_s1.json")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lam = [p for p in pts if p[0].startswith("λ") or p[0] == "No constraint"]
    xs = [100 * J(f)["core_cert"] for _, f in lam]; ys = [100 * J(f)["acc"] for _, f in lam]
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=5, label="Certificate loss (λ sweep)")
    # hand-placed label offsets (points) so clustered labels do not collide; thin leader lines
    off = {"No constraint": (4, 6), "λ = 0.03": (-46, -18), "λ = 0.1": (-6, -26), "λ = 0.3": (14, 14),
           "λ = 1": (-34, -6), "L1": (-26, 16), "ExCap": (6, 8)}
    lead = dict(arrowstyle="-", color=INK2, lw=0.5)
    for (lab, f), xv, yv in zip(lam, xs, ys):
        ax.annotate(lab, (xv, yv), textcoords="offset points", xytext=off[lab], fontsize=6.5, color=INK,
                    arrowprops=None if lab == "No constraint" else lead)
    for lab, f, col, mk in (("L1", "s2_shd_l1_l0_s1.json", YELLOW, "s"), ("ExCap", "s2_shd_clamp_l0_s1.json", AQUA, "D")):
        d = J(f); ax.scatter(100 * d["core_cert"], 100 * d["acc"], color=col, marker=mk, s=40, zorder=3, label=lab,
                              edgecolor=INK2, linewidth=0.5)
        ax.annotate(lab, (100 * d["core_cert"], 100 * d["acc"]), textcoords="offset points", xytext=off[lab],
                    fontsize=6.5, color=INK, arrowprops=lead)
    ax.set_xlabel("Provably silent core-steps, K = 4 (%)"); ax.set_ylabel("SHD test accuracy (%)")
    ax.set_xlim(-3, 72); ax.set_ylim(74, 85); ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("512-ALIF, seed 1", fontsize=8.5, loc="left", color=INK)
    fig.savefig(OUT + "fig4_tradeoff.png"); plt.close(fig)


# ---------------- Fig 5: mechanism (R_i distribution, margin vs horizon) ----------------
def fig_mechanism():
    import pilot_silence as ps
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
    ax = axs[0]
    Rs = {}
    for tag, lam in (("No constraint", "0"), ("Certificate loss", "0.1")):
        vals = []
        for s in (1, 2, 3):
            sd = torch.load(os.path.expanduser(f"~/research/models/p005_s{s}_l{lam}.pt"))
            W = (sd["wrec.weight"] * sd["mask"]).numpy(); vals.append(np.maximum(W, 0).sum(1))
        Rs[tag] = np.concatenate(vals)
    bins = np.linspace(0, 3.2, 49)
    ax.hist(Rs["No constraint"], bins, color=ORANGE, alpha=0.85, label="No constraint", edgecolor="white", linewidth=0.4)
    ax.hist(Rs["Certificate loss"], bins, color=BLUE, alpha=0.85, label="Certificate loss", edgecolor="white", linewidth=0.4)
    budget = (1 - ps.BETA) * ps.THETA
    ax.axvline(budget, color=INK, lw=1.2, ls="--"); ax.axvline(ps.THETA, color=INK2, lw=1.0, ls=":")
    ax.text(budget + 0.04, ax.get_ylim()[1] * 0.92, "(1−β)θ", fontsize=7, color=INK)
    ax.text(ps.THETA + 0.04, ax.get_ylim()[1] * 0.80, "θ", fontsize=7, color=INK2)
    ax.set_xlabel("Worst-case excitatory drive R_i"); ax.set_ylabel("Neurons (3 seeds)"); ax.legend(fontsize=6.5)
    ax.set_title("a  Excitatory drive is pushed below the budget", fontsize=8.5, loc="left", color=INK)
    ax = axs[1]; h = J("horizon001.json")
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2, [r["median_h_pos"] for r in h], w, color=BLUE, label="margin m_i > 0")
    b2 = ax.bar(x + w / 2, [r["median_h_neg"] for r in h], w, color=ORANGE, label="margin m_i ≤ 0")
    label_bars(ax, b1, "{:.0f}", 0.6); label_bars(ax, b2, "{:.0f}", 0.6)
    for xi, r in zip(x, h):
        ax.text(xi, 43, f"ρ = {r['spearman_margin_meanh']:.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [f"Seed {r['seed']}" for r in h]); ax.set_ylabel("Median certified horizon (steps)")
    ax.set_ylim(0, 56); ax.legend(fontsize=6.5, loc="upper center", ncol=2)
    ax.set_title("b  Margin predicts certified horizon", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(OUT + "fig5_mechanism.png"); plt.close(fig)


# ---------------- Fig 6: speed (engine) ----------------
S1C = {"L": [0, 5, 20, 100, 500],                                  # S1c log (ring-local, 8 cores, seed 1)
       "handshake": [0.474, 1.059, 2.643, 10.511, 49.845],
       "cert": [0.552, 0.784, 1.758, 6.803, 32.071],
       "control_cert": [0.569, 1.101, 2.625, 10.555, 49.780], "control_hs": [0.554, 1.083, 2.614, 10.542, 49.755]}


def parse_engine(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8-sig"):
        if line.startswith("RESULT"):
            kv = dict(p.split("=") for p in line.split()[1:])
            out.setdefault(kv["model"], {}).setdefault(kv["mode"], {})[int(kv["L_us"])] = float(kv["median_ms"])
    return out


def fig_speed():
    eng = parse_engine(R + "engines.txt")
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    L = np.array(S1C["L"]); xs = np.maximum(L, 1)
    sp = np.array(S1C["handshake"]) / np.array(S1C["cert"])
    spc = np.array(S1C["control_hs"]) / np.array(S1C["control_cert"])
    ax.plot(xs, sp, "-o", color=BLUE, lw=2, ms=5, label="Certificate loss (8×32 local)")
    ax.plot(xs, spc, "-o", color=ORANGE, lw=2, ms=5, label="No constraint (8×32 local)")
    for tag, col, mk, lab in (("clamp", AQUA, "D", "ExCap (8×32 local)"), ("l1", YELLOW, "s", "L1 (8×32 local)"),
                              ("dense_certified", MAGENTA, "^", "Certificate loss, 512-ALIF dense (8×64)")):
        if tag in eng and "cert" in eng[tag]:
            Ls = sorted(eng[tag]["cert"]); v = [eng[tag]["handshake"][l] / eng[tag]["cert"][l] for l in Ls]
            ax.plot(np.maximum(Ls, 1), v, "-" + mk, color=col, lw=1.5, ms=5, label=lab)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks([1, 5, 20, 100, 500], ["0", "5", "20", "100", "500"])
    ax.set_xlabel("Emulated interconnect latency (µs)"); ax.set_ylabel("Speed-up over local handshake (×)")
    ax.set_ylim(0.6, 2.45); ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Exact execution speed-up", fontsize=8.5, loc="left", color=INK)
    fig.savefig(OUT + "fig6_speed.png"); plt.close(fig)


# ---------------- Fig 1: concept schematic ----------------
def fig_concept():
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 2.6), sharex=True)
    steps = 8; lat = 0.6; comp = 0.4
    for ax, mode in zip(axs, ("handshake", "cert")):
        t = 0.0; y = 0
        for k in range(steps):
            ax.add_patch(plt.Rectangle((t, y - 0.25), comp, 0.5, facecolor=BLUE, edgecolor="white", linewidth=1.5))
            t += comp
            certified = mode == "cert" and k % 4 != 3
            if not certified:
                ax.add_patch(plt.Rectangle((t, y - 0.12), lat, 0.24, color=GRID, ec=INK2, lw=0.4))
                t += lat
        ax.set_xlim(0, steps * (comp + lat) + 0.2); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.grid(False); ax.spines["left"].set_visible(False); ax.set_xticks([])
        ax.text(0, 0.42, "Local handshake: wait for neighbours every step" if mode == "handshake" else
                "Certified: neighbour certified silent for 3 steps → no waiting", fontsize=7.5, color=INK)
    axs[1].set_xlabel("Wall-clock time (schematic). Blue = compute one step; grey = wait for neighbour data.")
    fig.tight_layout(); fig.savefig(OUT + "fig1_concept.png"); plt.close(fig)


# ---------------- Fig S1: SWEEP-001 (validation, 22 configs) ----------------
def fig_sweep():
    import pandas as pd
    T = pd.read_csv(R + "sweep_table.csv")
    fam = lambda t: "Reference" if t.startswith("ref") else ("A: sink hubs" if t.startswith("A_") else "C: activity-weighted")
    T["fam"] = T.tag.map(fam)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for f, col, mk in (("Reference", BLUE, "o"), ("A: sink hubs", ORANGE, "s"), ("C: activity-weighted", AQUA, "^")):
        d = T[T.fam == f]
        ax.scatter(d.cert, d.acc_val, color=col, marker=mk, s=34, label=f, edgecolor=INK2, linewidth=0.4, zorder=3)
    ctrl = float(T[T.tag == "ref_ctrl"].acc_val.iloc[0])
    ax.axhline(ctrl - 1.0, color=INK2, lw=0.8, ls="--")
    ax.text(1, ctrl - 1.0 + 0.12, "selection threshold (control − 1 pt)", fontsize=6.5, color=INK2)
    for t, lab, dx, dy in (("ref_clamp", "ExCap (selected)", -60, 8), ("ref_ctrl", "control", 4, -10),
                           ("ref_cert03", "certificate loss", 4, -12), ("A_h4_cert03", "4 sink hubs", -46, -14)):
        r = T[T.tag == t].iloc[0]
        ax.annotate(lab, (r.cert, r.acc_val), textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_xlabel("Validation: provably silent waited-on core-steps, K = 4 (%)"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlim(-3, 72); ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("SWEEP-001: 22 configurations (seed 1)", fontsize=8.5, loc="left", color=INK)
    fig.savefig(OUT + "figS1_sweep.png"); plt.close(fig)


if __name__ == "__main__":
    fig_concept(); fig_gap(); m = fig_methods(); fig_tradeoff(); fig_mechanism(); fig_speed(); fig_sweep()
    json.dump({k: v.tolist() for k, v in m.items()}, open(R + "small_methods_table.json", "w"), indent=1)
    print("figures written to", OUT)
