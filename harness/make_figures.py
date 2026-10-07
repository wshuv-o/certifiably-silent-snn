"""Build the manuscript figures.

Two figures.

Figure 1 is the mechanism: what the delay decomposition says, what the certificate compares, and
what it licenses at run time. Panels (a) and (c) are diagrams of the algebra and the protocol.
Panel (b) is computed from the model equations with the measured R_short values -- the worst-case
reach is beta^k v_0 + R (1 - beta^k)/(1 - beta), evaluated at beta = exp(-1/2), theta = 1.

Figure 2 is the evidence, every value transcribed from a recorded experiment; the comment above each
block names the table it came from. Nothing is interpolated, smoothed or synthesised.

Style follows the journal's line-art conventions: serif type matching the body text, thin rules, no
panel titles where the caption carries them, markers that stay distinguishable in greyscale, and
vector output with embedded Type 42 fonts. Percent signs are written bare because matplotlib's
mathtext is used rather than a LaTeX backend.

    python harness/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
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

BETA = np.exp(-0.5)          # membrane decay, Delta t / tau_m = 1/2
THETA = 1.0                  # base threshold
BUDGET = (1.0 - BETA) * THETA   # 0.3935


def panel_label(ax, s, dx=-0.17, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
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
    fig = plt.figure(figsize=(6.3, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.74, 1.0], width_ratios=[1.0, 1.28],
                          hspace=0.46, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------------- (a) the decomposition ---
    # A certificate rooted after step t+1 at horizon k reads the step tau - d = t+1+k-d.
    # d >= k lands at or before t+1, which is already computed; d < k lands inside the horizon.
    ax = ax_a
    steps = np.arange(-6, 6)          # offsets from t
    K = 4
    tau = 5                           # the step under consideration, t+1+K with K = 4
    for o in steps:
        known = o <= 1
        ax.add_patch(Rectangle((o - 0.34, -0.26), 0.68, 0.52,
                               facecolor=K_PALE if known else "white",
                               edgecolor=K_BLACK if known else K_RED,
                               lw=0.7, ls="-" if known else (0, (2.5, 1.5)), zorder=3))
    ax.add_patch(Rectangle((tau - 0.34, -0.26), 0.68, 0.52, facecolor="white",
                           edgecolor=K_RED, lw=1.4, zorder=4))

    ax.plot([1.5, 1.5], [-1.5, 2.02], color=K_DARK, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(1.42, 2.00, "computed through $t+1$", fontsize=6.6, ha="right", va="top",
            color=K_DARK)
    ax.text(1.62, 2.00, "certificate horizon $K=4$", fontsize=6.6, ha="left", va="top",
            color=K_RED)
    ax.annotate("", xy=(5.4, 1.70), xytext=(1.6, 1.70),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", lw=0.7, color=K_RED))

    # The three taps feeding the step tau. arc3's rad is a fraction of the span, so the apex
    # height is set explicitly and rad derived from it; otherwise the long d=8 arc overshoots.
    for d, apex, col, ls, tag in ((8, 1.12, K_BLACK, "-", "exact"),
                                  (4, 0.70, K_BLACK, "-", "exact"),
                                  (2, 0.34, K_RED, (0, (3, 1.8)), "bounded")):
        src = tau - d
        rad = 2.0 * apex / float(d)
        ax.add_patch(FancyArrowPatch((src, 0.30), (tau, 0.30),
                                     connectionstyle="arc3,rad=" + str(-rad),
                                     arrowstyle="-|>,head_width=1.5,head_length=3.0",
                                     lw=0.9, color=col, ls=ls, zorder=5,
                                     shrinkA=1, shrinkB=2))
        ax.text(src + 0.27 * d, 0.30 + 0.80 * apex + 0.06, "$d=" + str(d) + "$, " + tag,
                fontsize=6.5, ha="center", va="bottom", color=col)

    ax.text(-6.6, 0.0, "steps", fontsize=7, ha="right", va="center", color=K_DARK)
    for o, lab in ((-6, "$t-6$"), (0, "$t$"), (1, "$t{+}1$"), (5, "$\\tau = t{+}5$")):
        ax.text(o, -0.52, lab, fontsize=6.6, ha="center", va="top",
                color=K_RED if o == 5 else K_DARK)
    ax.text(5.0, -1.12, "bounded drive $R_{\\mathrm{short}} = \\sum_{d<K} R^{(d)}$",
            fontsize=7, ha="center", va="top", color=K_RED)
    ax.set_xlim(-7.4, 6.4)
    ax.set_ylim(-1.30, 2.06)
    ax.axis("off")
    panel_label(ax, "(a)", dx=-0.015, dy=1.10)

    # ---------------------------------------------------------------- (c) what it licenses ----
    # schematic of the two protocols; the saving shown is qualitative, not a measured ratio
    ax = ax_c
    comp, wait = 0.62, 0.40
    for row, (name, per_step_wait) in enumerate((("certificate", False),
                                                 ("lock-step", True))):
        y = row * 1.0
        x = 0.0
        if not per_step_wait:                       # one wait, then run the horizon out
            ax.add_patch(Rectangle((x, y - 0.22), wait, 0.44, facecolor=K_PALE,
                                   edgecolor=K_GREY, lw=0.6, hatch="////", zorder=3))
            x += wait
        for _ in range(5):
            if per_step_wait:
                ax.add_patch(Rectangle((x, y - 0.22), wait, 0.44, facecolor=K_PALE,
                                       edgecolor=K_GREY, lw=0.6, hatch="////", zorder=3))
                x += wait
            ax.add_patch(Rectangle((x, y - 0.22), comp, 0.44, facecolor="white",
                                   edgecolor=K_BLACK, lw=0.7, zorder=3))
            x += comp
        ax.text(-0.12, y, name, fontsize=7, ha="right", va="center", color=K_DARK)
        ax.plot([x, x], [y - 0.34, y + 0.34], color=K_RED if row == 0 else K_DARK,
                lw=0.9, zorder=4)
    ax.annotate("", xy=(3.50, 1.52), xytext=(5.10, 1.52),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=K_RED,
                                shrinkA=0, shrinkB=0))
    ax.text(4.30, 1.60, "time saved", fontsize=6.4, ha="center", color=K_RED)
    leg = [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=K_BLACK, lw=0.7),
           Rectangle((0, 0), 1, 1, facecolor=K_PALE, edgecolor=K_GREY, lw=0.6, hatch="////")]
    ax.legend(leg, ["compute a step", "blocked on peer"], loc="lower center",
              bbox_to_anchor=(0.46, -0.30), ncol=2, handlelength=1.4, handletextpad=0.5,
              columnspacing=1.0, fontsize=6.6)
    ax.set_xlim(-1.9, 5.5)
    ax.set_ylim(-0.75, 1.88)
    ax.axis("off")
    panel_label(ax, "(c)", dx=-0.03, dy=1.12)

    # ---------------------------------------------------------------- (b) the budget ----------
    # worst-case reach from a measured v(t+1) = 0 under a drive bounded by R at every step
    ax = ax_b
    ks = np.arange(0, 9)
    for R, col, mk, lab in ((4.968, K_BLACK, "s", "control"),
                            (0.275, K_BLUE, "o", "constrained")):
        reach = R * (1.0 - BETA ** ks) / (1.0 - BETA)
        ax.plot(ks, np.maximum(reach, 1e-3), marker=mk, color=col, mfc="white", mec=col,
                mew=0.8, label=lab + "  ($R_{\\mathrm{short}}=" + ("%.3f" % R) + "$)", zorder=3)
    ax.axhline(THETA, color=K_RED, lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax.text(8.0, THETA * 1.22, "threshold $\\theta$", fontsize=6.5, color=K_RED, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.04, 40)
    ax.set_xlim(-0.3, 8.3)
    ax.set_xlabel("horizon step $k$")
    ax.set_ylabel("worst-case reach")
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.04), handletextpad=0.4,
              labelspacing=0.25, handlelength=1.4, fontsize=6.3)
    panel_label(ax, "(b)", dx=-0.30, dy=1.12)

    save(fig, "fig1_mechanism")


# =============================================================================================
# Figure 2 -- the evidence.
#   (a,b) manuscript table 1 (tab:governing), speaker-disjoint validation, seed 1.
#   (c,d) manuscript table 4 (tab:mech); points in (d) are the four SSC test seeds.
#   (e)   manuscript table 5 (tab:speed), shared-memory engine, 80/80 bit-identical.
#   (f)   manuscript table 6 (tab:tcp), two-process TCP engine, 54/54 bit-identical.
# =============================================================================================
def figure2():
    rows = [
        (1,  0.0, 4.840,  4.84, False),   # unit delay {1}
        (2,  2.7, 5.983, 12.30, False),   # {1,2,4,8}
        (3,  6.0, 4.968, 15.27, False),   # {2,4,8} control
        (4, 94.9, 0.275, 12.76, True),    # {2,4,8} constrained
        (5, 97.4, 1.595,  7.48, False),   # learnable delays, control
        (6, 98.0, 0.392,  7.00, True),    # learnable delays, constrained
    ]
    pct = np.array([r[1] for r in rows])
    rs = np.array([r[2] for r in rows])
    tot = np.array([r[3] for r in rows])
    cons = [r[4] for r in rows]
    off_a = {1: (0, -12), 2: (8, -3), 3: (0, 7), 4: (0, -12), 5: (8, -3), 6: (7, 4)}
    ha_a = {1: "center", 2: "left", 3: "center", 4: "center", 5: "left", 6: "left"}
    off_b = {1: (8, -3), 2: (0, -12), 3: (0, 7), 4: (8, -3), 5: (8, -2), 6: (-8, -2)}
    ha_b = {1: "left", 2: "center", 3: "center", 4: "left", 5: "left", 6: "right"}

    shd_c = np.array([4.97, 4.90, 5.40])
    shd_k = np.array([0.28, 6.03, 6.46])
    ssc_c = np.array([5.73, 6.12, 7.19])
    ssc_seeds = np.array([[0.29, 8.56, 9.98], [0.29, 8.54, 9.84],
                          [0.30, 7.65, 8.89], [0.29, 8.21, 9.42]])
    ssc_k = ssc_seeds.mean(axis=0)

    cores = np.array([4, 8, 16, 32])
    lats = [0, 5, 20, 100, 500]
    sm = {0: [0.93, 0.90, 0.79, 1.44], 5: [0.95, 0.88, 0.79, 1.35],
          20: [0.91, 0.90, 0.78, 1.30], 100: [1.07, 1.02, 1.05, 1.25],
          500: [1.11, 1.11, 1.11, 1.49]}
    ctrl32 = [0.99, 0.98, 0.83, 0.98, 0.99]

    L = np.array([0, 200, 400, 800, 1600])
    hs = np.array([13.08, 17.22, 26.71, 47.09, 86.50])
    ce = np.array([17.59, 19.73, 26.03, 41.67, 73.44])
    sk = np.array([17.57, 19.71, 28.01, 45.03, 79.28])

    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4),
                             gridspec_kw={"wspace": 0.42, "hspace": 0.58})

    # ---- (a,b) the governing relation and its control ---------------------------------------
    for ax, x, xl, off, hal, lab in ((axes[0, 0], rs, "$R_{\\mathrm{short}}$", off_a, ha_a, "(a)"),
                                     (axes[0, 1], tot, "total excitation $R$", off_b, ha_b, "(b)")):
        for i, r in enumerate(rows):
            ax.plot(x[i], pct[i], marker="o" if cons[i] else "s", ms=4.5, linestyle="none",
                    mfc=K_BLUE if cons[i] else "white",
                    mec=K_BLUE if cons[i] else K_BLACK, mew=0.8, zorder=3)
            ax.annotate(str(r[0]), (x[i], pct[i]), textcoords="offset points",
                        xytext=off[r[0]], ha=hal[r[0]], fontsize=6.5, color=K_DARK, zorder=4)
        ax.set_xlabel(xl)
        ax.set_ylim(-22, 120)
        ax.set_ylabel("certified (% of oracle)")
        panel_label(ax, lab, dx=-0.34)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlim(0.19, 11)
    axes[0, 0].axvline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    axes[0, 0].text(0.355, 55, "budget", fontsize=6.0, color=K_RED, ha="right", rotation=90,
                    va="center")
    axes[0, 1].set_xlim(3.2, 17.6)

    # ---- (c) certification against accuracy, the cost question ------------------------------
    # manuscript tables 2 and 3: SHD three test seeds, SSC four test seeds
    ax = axes[0, 2]
    shd = [(87.28, 87.90), (86.75, 86.62), (88.03, 88.25)]
    ssc = [(65.56, 68.66), (65.25, 68.06), (65.24, 68.33), (64.61, 67.75)]
    for pairs, col, mk, lab in ((shd, K_BLUE, "o", "SHD"), (ssc, K_RED, "^", "SSC")):
        for c, k in pairs:
            ax.plot([c], [k], marker=mk, ms=4.5, mfc="white", mec=col, mew=0.9,
                    linestyle="none", zorder=3, label=lab)
            lab = None
    lo, hi = 63.5, 89.5
    ax.plot([lo, hi], [lo, hi], color=K_GREY, lw=0.7, ls=":", zorder=2)
    ax.text(hi - 0.6, lo + 1.6, "no change", fontsize=6.0, color=K_GREY, ha="right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("control accuracy (%)")
    ax.set_ylabel("constrained accuracy (%)")
    ax.legend(loc="upper left", handletextpad=0.3, labelspacing=0.2, fontsize=6.3)
    panel_label(ax, "(c)", dx=-0.34)

    # ---- (d,e) the mechanism ----------------------------------------------------------------
    w, xs = 0.34, np.arange(3)
    for ax, ctrl, cons_r, seeds, head, lab in (
            (axes[1, 0], shd_c, shd_k, None, "SHD, total $-16$%", "(d)"),
            (axes[1, 1], ssc_c, ssc_k, ssc_seeds, "SSC, total $-1$%", "(e)")):
        ax.bar(xs - w / 2, ctrl, w, color="white", edgecolor=K_BLACK, lw=0.7,
               label="control", zorder=3)
        ax.bar(xs + w / 2, cons_r, w, color=K_GREY, edgecolor=K_BLACK, lw=0.7,
               label="constrained", zorder=3)
        if seeds is not None:
            for s in seeds:
                ax.plot(xs + w / 2, s, marker="o", ms=2.2, mfc=K_BLACK, mec="none",
                        linestyle="none", zorder=5)
        ax.axhline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=4)
        ax.annotate("", xy=(w / 2, cons_r[0] + 1.2), xytext=(-w / 2, ctrl[0] + 0.3),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=K_RED,
                                    connectionstyle="arc3,rad=-0.3"), zorder=6)
        ax.text(0.34, ctrl[0] + 1.8, "$-94$%", fontsize=6.5, color=K_RED, ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(["$2$", "$4$", "$8$"])
        ax.set_xlabel("synaptic delay $d$")
        ax.set_ylabel("drive $R^{(d)}$")
        ax.set_ylim(0, 11.6)
        ax.set_title(head, fontsize=6.8, color=K_DARK, pad=3)
        panel_label(ax, lab, dx=-0.34, dy=1.13)
    axes[1, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), handletextpad=0.45,
                      labelspacing=0.22, handlelength=1.1, fontsize=6.3)

    # ---- (f) execution ----------------------------------------------------------------------
    ax = axes[1, 2]
    shades = ["#000000", "#555555", "#8a8a8a"]
    for lat, mk, cl in zip([0, 100, 500], ["o", "D", "v"], shades):
        ax.plot(cores, sm[lat], marker=mk, color=cl, mfc="white", mec=cl, mew=0.8,
                label="$L=" + str(lat) + "\\,\\mu$s", zorder=3)
    ax.plot([32] * len(ctrl32), ctrl32, marker="x", color=K_GREY, mew=0.9, ms=3.6,
            linestyle="none", zorder=4)
    ax.text(30, 0.80, "control", fontsize=6.0, color=K_GREY, ha="right")
    ax.axhline(1.0, color=K_GREY, lw=0.7, ls=":", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    ax.set_xlabel("cores (shared memory)")
    ax.set_ylabel("speed-up")
    ax.set_ylim(0.70, 1.62)
    ax.legend(loc="upper left", handletextpad=0.35, labelspacing=0.18, handlelength=1.1,
              fontsize=6.0)
    panel_label(ax, "(f)", dx=-0.34)

    # two-process result as an inset-style twin on the same panel is too dense; keep it in the
    # tables and report the break-even in the text.
    save(fig, "fig2_evidence")


if __name__ == "__main__":
    figure1()
    figure2()
    print("\nfigures in " + os.path.relpath(OUT))
