"""Build the manuscript figures.

Two figures.

Figure 1 is the mechanism: what the delay decomposition says, where a trained network puts its
excitation in time, what the certificate compares, and what holding one licenses at run time.
Panels (a) and (d) are diagrams of the algebra and the protocol. Panel (b) is the measured delay
distribution of the learned-delay checkpoint. Panel (c) is computed from the model equations with
the measured R_short values: the worst-case reach is beta^k v + R (1 - beta^k)/(1 - beta), at
beta = exp(-1/2) and theta = 1.

Figure 2 is the evidence, every value transcribed from a recorded experiment; the comment above each
block names the table it came from. Nothing is interpolated, smoothed or synthesised.

Text is kept off the panels wherever the caption can carry it, so nothing collides at print size.
Style follows the journal's line-art conventions: serif type matching the body text, thin rules,
markers that stay distinguishable in greyscale, vector output with Type 42 fonts.

    python harness/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

K_BLACK = "#000000"
K_DARK = "#333333"
K_GREY = "#8a8a8a"
K_PALE = "#dcdcdc"
K_BLUE = "#1f4e79"
K_RED = "#9c2b26"

BETA = np.exp(-0.5)                 # membrane decay, Delta t / tau_m = 1/2
THETA = 1.0                         # base threshold
BUDGET = (1.0 - BETA) * THETA       # 0.3935


def panel_label(ax, s, dx=-0.17, dy=1.07):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="top", ha="left")


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p + ".pdf")
    fig.savefig(p + ".png", dpi=400)
    plt.close(fig)
    print("wrote " + os.path.relpath(p) + ".pdf")


# =============================================================================================
# Figure 1 -- the mechanism.
# =============================================================================================
def figure1():
    fig = plt.figure(figsize=(6.3, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.58, 1.0], hspace=0.48, wspace=0.48)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # ---------------------------------------------------------------- (a) the decomposition ---
    # A certificate at horizon k reads step tau - d = t+1+k-d: already computed when d >= k,
    # still inside the horizon when d < k.
    ax = ax_a
    tau = 5
    for o in np.arange(-6, 6):
        known = o <= 1
        ax.add_patch(Rectangle((o - 0.34, -0.26), 0.68, 0.52,
                               facecolor=K_PALE if known else "white",
                               edgecolor=K_BLACK if known else K_RED,
                               lw=0.7, ls="-" if known else (0, (2.5, 1.5)), zorder=3))
    ax.add_patch(Rectangle((tau - 0.34, -0.26), 0.68, 0.52, facecolor="white",
                           edgecolor=K_RED, lw=1.4, zorder=4))
    ax.plot([1.5, 1.5], [-0.62, 1.86], color=K_DARK, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(1.40, 1.84, "computed", fontsize=6.4, ha="right", va="top", color=K_DARK)
    ax.text(1.60, 1.84, "horizon $K=4$", fontsize=6.4, ha="left", va="top", color=K_RED)
    ax.annotate("", xy=(5.4, 1.56), xytext=(1.6, 1.56),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", lw=0.7, color=K_RED))

    # arc3's rad is a fraction of the span, so set the apex and derive rad from it
    for d, apex, col, ls in ((8, 1.04, K_BLACK, "-"), (4, 0.66, K_BLACK, "-"),
                             (2, 0.30, K_RED, (0, (3, 1.8)))):
        src = tau - d
        ax.add_patch(FancyArrowPatch((src, 0.30), (tau, 0.30),
                                     connectionstyle="arc3,rad=" + str(-2.0 * apex / d),
                                     arrowstyle="-|>,head_width=1.5,head_length=3.0",
                                     lw=0.9, color=col, ls=ls, zorder=5, shrinkA=1, shrinkB=2))
        ax.text(src + 0.26 * d, 0.30 + 0.98 * apex + 0.05, "$d=" + str(d) + "$",
                fontsize=6.4, ha="center", va="bottom", color=col)
    for o, lab in ((0, "$t$"), (1, "$t{+}1$"), (5, "$\\tau$")):
        ax.text(o, -0.38, lab, fontsize=6.4, ha="center", va="top",
                color=K_RED if o == 5 else K_DARK)
    ax.set_xlim(-6.9, 6.1)
    ax.set_ylim(-0.66, 1.90)
    ax.axis("off")
    panel_label(ax, "(a)", dx=-0.012, dy=1.12)

    # ---------------------------------------------------------------- (b) learned delays -----
    # Excitatory mass over delay, from the corrected learnable-delay checkpoints dx_b_fix_{ctrl,cert}:
    # D = 2 + 6 sigmoid(draw), binned in 24 bins and weighted by max(0, w). A working learner
    # saturates both rails of the sigmoid; the penalty evacuates the short one.
    edges = np.linspace(2.0, 8.0, 25)
    mass_ctrl = np.array([0.33306, 0.02929, 0.01741, 0.01551, 0.01286, 0.00918, 0.00848, 0.00954,
                          0.00988, 0.00805, 0.00819, 0.00952, 0.00909, 0.00757, 0.00762, 0.00996,
                          0.00975, 0.00894, 0.01015, 0.01447, 0.01844, 0.01994, 0.03204, 0.38106])
    mass_cert = np.array([0.03726, 0.00325, 0.00169, 0.00207, 0.00167, 0.00248, 0.00249, 0.00805,
                          0.01761, 0.01856, 0.01773, 0.02399, 0.01855, 0.01442, 0.01400, 0.01598,
                          0.01556, 0.01219, 0.01359, 0.02397, 0.02595, 0.03039, 0.05135, 0.62721])
    ax = ax_b
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.axvspan(1.9, 4.0, color=K_RED, alpha=0.09, lw=0, zorder=1)
    ax.step(centres, mass_ctrl * 100.0, where="mid", color=K_BLACK, lw=0.9,
            label="control", zorder=3)
    ax.step(centres, mass_cert * 100.0, where="mid", color=K_BLUE, lw=0.9,
            ls=(0, (3, 1.4)), label="constrained", zorder=4)
    ax.axvline(4.0, color=K_RED, lw=0.9, ls=(0, (4, 2)), zorder=5)
    ax.text(3.88, 0.30, "$d<K$", fontsize=6.4, color=K_RED, ha="right", va="center")
    ax.set_yscale("log")
    ax.set_xlim(1.9, 8.1)
    ax.set_ylim(0.1, 160)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("learned delay $d$")
    ax.set_ylabel("excitatory mass (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.06), handletextpad=0.35,
              labelspacing=0.15, handlelength=1.3, fontsize=6.2, borderpad=0.25)
    panel_label(ax, "(b)", dx=-0.36)

    # ---------------------------------------------------------------- (c) the budget ---------
    ax = ax_c
    ks = np.arange(0, 9)
    for R, col, mk, lab in ((4.968, K_BLACK, "s", "control"),
                            (0.275, K_BLUE, "o", "constrained")):
        ax.plot(ks, np.maximum(R * (1.0 - BETA ** ks) / (1.0 - BETA), 1e-3),
                marker=mk, color=col, mfc="white", mec=col, mew=0.8, label=lab, zorder=3)
    ax.axhline(THETA, color=K_RED, lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax.text(8.2, 1.3, "$\\theta$", fontsize=7.5, color=K_RED, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 40)
    ax.set_xlim(-0.3, 8.3)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("horizon step $k$")
    ax.set_ylabel("worst-case reach")
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.03), handletextpad=0.35,
              labelspacing=0.2, handlelength=1.2)
    panel_label(ax, "(c)", dx=-0.36)

    # ---------------------------------------------------------------- (d) what it licenses ---
    ax = ax_d
    comp, wait = 0.62, 0.40
    for row, (name, per_step) in enumerate((("certificate", False), ("lock-step", True))):
        y, x = row * 1.0, 0.0
        if not per_step:
            ax.add_patch(Rectangle((x, y - 0.21), wait, 0.42, facecolor=K_PALE,
                                   edgecolor=K_GREY, lw=0.6, hatch="////", zorder=3))
            x += wait
        for _ in range(5):
            if per_step:
                ax.add_patch(Rectangle((x, y - 0.21), wait, 0.42, facecolor=K_PALE,
                                       edgecolor=K_GREY, lw=0.6, hatch="////", zorder=3))
                x += wait
            ax.add_patch(Rectangle((x, y - 0.21), comp, 0.42, facecolor="white",
                                   edgecolor=K_BLACK, lw=0.7, zorder=3))
            x += comp
        ax.text(-0.14, y, name, fontsize=6.6, ha="right", va="center", color=K_DARK)
        ax.plot([x, x], [y - 0.32, y + 0.32], color=K_RED if row == 0 else K_DARK, lw=0.9,
                zorder=4)
    ax.annotate("", xy=(3.50, 1.52), xytext=(5.10, 1.52),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=K_RED, shrinkA=0, shrinkB=0))
    ax.text(4.30, 1.60, "saved", fontsize=6.4, ha="center", color=K_RED)
    leg = [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=K_BLACK, lw=0.7),
           Rectangle((0, 0), 1, 1, facecolor=K_PALE, edgecolor=K_GREY, lw=0.6, hatch="////")]
    ax.legend(leg, ["compute", "blocked"], loc="lower center", bbox_to_anchor=(0.46, -0.16),
              ncol=2, handlelength=1.2, handletextpad=0.4, columnspacing=0.9, fontsize=6.4)
    ax.set_xlim(-2.4, 5.5)
    ax.set_ylim(-0.66, 1.88)
    ax.axis("off")
    panel_label(ax, "(d)", dx=-0.04, dy=1.12)

    save(fig, "fig1_mechanism")


# =============================================================================================
# Figure 2 -- the evidence.
#   (a,b) manuscript table 1 (tab:governing), speaker-disjoint validation.
#   (c)   manuscript tables 2 and 3, held-out test seeds.
#   (d,e) manuscript table 4 (tab:mech); points in (e) are the four SSC test seeds.
#   (f)   manuscript table 5 (tab:speed), shared-memory engine, 240/240 bit-identical.
# =============================================================================================
def figure2():
    rows = [
        (1,  0.0, 7.905,  7.91, False),   # unit delay {1}
        (2,  3.8, 8.741, 18.40, False),   # {1,2,4,8}
        (3,  6.3, 4.813, 14.98, False),   # {2,4,8} control
        (4, 95.0, 0.261, 12.64, True),    # {2,4,8} constrained
        (5,  1.9, 3.058,  7.36, False),   # learnable delays, control (corrected learner, DCLS-002)
        (6, 95.5, 0.246,  4.85, True),    # learnable delays, constrained
    ]
    pct = np.array([r[1] for r in rows])
    rs = np.array([r[2] for r in rows])
    tot = np.array([r[3] for r in rows])
    cons = [r[4] for r in rows]
    off_a = {1: (0, -11), 2: (8, -3), 3: (0, 7), 4: (0, -11), 5: (8, -3), 6: (7, 4)}
    ha_a = {1: "center", 2: "left", 3: "center", 4: "center", 5: "left", 6: "left"}
    off_b = {1: (8, -3), 2: (0, -11), 3: (0, 7), 4: (8, -3), 5: (8, -2), 6: (-8, -2)}
    ha_b = {1: "left", 2: "center", 3: "center", 4: "left", 5: "left", 6: "right"}

    shd_c = np.array([4.97, 4.90, 5.40])
    shd_k = np.array([0.28, 6.03, 6.46])
    ssc_c = np.array([5.73, 6.12, 7.19])
    ssc_seeds = np.array([[0.29, 8.56, 9.98], [0.29, 8.54, 9.84],
                          [0.30, 7.65, 8.89], [0.29, 8.21, 9.42]])
    ssc_k = ssc_seeds.mean(axis=0)

    cores = np.array([4, 8, 16, 32])
    # SPEED-DIAG-001, idle machine, medians of three repeats (supersedes an earlier contaminated run)
    sm = {0: [0.93, 0.87, 0.81, 0.73], 100: [1.06, 1.07, 1.05, 1.07],
          500: [1.11, 1.12, 1.11, 1.14]}
    ctrl32 = [0.76, 0.82, 0.84, 0.97, 0.96]   # SPEED-DIAG-001; control never exceeds 1.0

    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.2),
                             gridspec_kw={"wspace": 0.46, "hspace": 0.62})

    for ax, x, xl, off, hal, lab in ((axes[0, 0], rs, "$R_{\\mathrm{short}}$", off_a, ha_a, "(a)"),
                                     (axes[0, 1], tot, "total excitation $R$", off_b, ha_b, "(b)")):
        for i, r in enumerate(rows):
            ax.plot(x[i], pct[i], marker="o" if cons[i] else "s", ms=4.5, linestyle="none",
                    mfc=K_BLUE if cons[i] else "white",
                    mec=K_BLUE if cons[i] else K_BLACK, mew=0.8, zorder=3)
            ax.annotate(str(r[0]), (x[i], pct[i]), textcoords="offset points",
                        xytext=off[r[0]], ha=hal[r[0]], fontsize=6.5, color=K_DARK, zorder=4)
        ax.set_xlabel(xl)
        ax.set_ylim(-24, 122)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("certified (% of oracle)")
        panel_label(ax, lab, dx=-0.36)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlim(0.19, 11)
    axes[0, 0].axvline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    axes[0, 1].set_xlim(3.2, 17.6)

    # ---- (c) the accuracy cost --------------------------------------------------------------
    ax = axes[0, 2]
    shd = [(87.28, 87.90), (86.75, 86.62), (88.03, 88.25)]
    ssc = [(65.56, 68.66), (65.25, 68.06), (65.24, 68.33), (64.61, 67.75)]
    for pairs, col, mk, lab in ((shd, K_BLUE, "o", "SHD"), (ssc, K_RED, "^", "SSC")):
        ax.plot([c for c, _ in pairs], [k for _, k in pairs], marker=mk, ms=4.5, mfc="white",
                mec=col, mew=0.9, linestyle="none", label=lab, zorder=3)
    lo, hi = 63.5, 89.5
    ax.plot([lo, hi], [lo, hi], color=K_GREY, lw=0.7, ls=":", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([65, 75, 85])
    ax.set_yticks([65, 75, 85])
    ax.set_xlabel("control accuracy (%)")
    ax.set_ylabel("constrained (%)")
    ax.legend(loc="upper left", handletextpad=0.3, labelspacing=0.18)
    panel_label(ax, "(c)", dx=-0.36)

    # ---- (d,e) the mechanism ----------------------------------------------------------------
    w, xs = 0.34, np.arange(3)
    for ax, ctrl, cons_r, seeds, head, lab in (
            (axes[1, 0], shd_c, shd_k, None, "SHD", "(d)"),
            (axes[1, 1], ssc_c, ssc_k, ssc_seeds, "SSC", "(e)")):
        ax.bar(xs - w / 2, ctrl, w, color="white", edgecolor=K_BLACK, lw=0.7,
               label="control", zorder=3)
        ax.bar(xs + w / 2, cons_r, w, color=K_GREY, edgecolor=K_BLACK, lw=0.7,
               label="constrained", zorder=3)
        if seeds is not None:
            for s in seeds:
                ax.plot(xs + w / 2, s, marker="o", ms=2.2, mfc=K_BLACK, mec="none",
                        linestyle="none", zorder=5)
        ax.axhline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=4)
        ax.annotate("", xy=(w / 2, cons_r[0] + 1.1), xytext=(-w / 2, ctrl[0] + 0.3),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=K_RED,
                                    connectionstyle="arc3,rad=-0.3"), zorder=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["2", "4", "8"])
        ax.set_xlabel("synaptic delay $d$")
        ax.set_ylabel("drive $R^{(d)}$")
        ax.set_ylim(0, 11.4)
        ax.set_title(head, fontsize=6.8, color=K_DARK, pad=3)
        panel_label(ax, lab, dx=-0.36, dy=1.14)
    axes[1, 0].legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.4,
                      labelspacing=0.2, handlelength=1.0)

    # ---- (f) execution ----------------------------------------------------------------------
    ax = axes[1, 2]
    for lat, mk, cl in zip([0, 100, 500], ["o", "D", "v"], ["#000000", "#555555", "#8a8a8a"]):
        ax.plot(cores, sm[lat], marker=mk, color=cl, mfc="white", mec=cl, mew=0.8,
                label="$" + str(lat) + "\\,\\mu$s", zorder=3)
    ax.plot([32] * len(ctrl32), ctrl32, marker="x", color=K_GREY, mew=0.9, ms=3.6,
            linestyle="none", zorder=4)
    ax.axhline(1.0, color=K_GREY, lw=0.7, ls=":", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    ax.set_xlabel("cores")
    ax.set_ylabel("speed-up")
    ax.set_ylim(0.70, 1.62)
    ax.legend(loc="upper left", handletextpad=0.3, labelspacing=0.15, handlelength=1.0,
              title="latency", title_fontsize=6.2)
    panel_label(ax, "(f)", dx=-0.36)

    save(fig, "fig2_evidence")


if __name__ == "__main__":
    figure1()
    figure2()
    print("\nfigures in " + os.path.relpath(OUT))
