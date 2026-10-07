"""Build the manuscript figures from measured results.

Every number here is transcribed from a recorded experiment and the comment above each block names
the table or run it came from, so a figure traces back to the run that produced it. Nothing is
interpolated, smoothed or synthesised.

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
K_BLUE = "#1f4e79"
K_RED = "#9c2b26"
BUDGET = 0.3935          # (1 - beta) * theta with beta = exp(-1/2), theta = 1


def panel_label(ax, s, dx=-0.17, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p + ".pdf")
    fig.savefig(p + ".png", dpi=400)
    plt.close(fig)
    print("wrote " + os.path.relpath(p) + ".pdf")


# ---------------------------------------------------------------------------------------------
# Figure 1. The delay decomposition, and the horizon it permits.
#   (a) evaluates the condition d >= k exactly; it is a statement of the algebra, not a sketch.
#   (b,c) RECERT-001 (research/N3_SCALEUP_PLAN.md): identical weights re-certified at three
#         horizons, 18 runs, 0 violations. Only K changes.
# ---------------------------------------------------------------------------------------------
def figure1():
    delays = [2, 4, 8]
    ks = np.arange(1, 9)

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2),
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.0], "wspace": 0.46})

    # ---- (a) which taps a horizon binds -----------------------------------------------------
    ax = axes[0]
    # the staircase is the line d = k: below it a tap arrives from a step already computed
    bx, by = [], []
    for xi, d in enumerate(delays, start=1):
        bx += [xi - 0.5, xi + 0.5]
        by += [min(d, 8) + 0.5, min(d, 8) + 0.5]
    ax.step(bx, by, where="post", color=K_DARK, lw=0.9, zorder=4)
    for xi, d in enumerate(delays, start=1):
        for k in ks:
            bounded = d < k
            ax.plot(xi, k, marker="s", ms=5.5, linestyle="none", zorder=3,
                    mfc=K_RED if bounded else "white",
                    mec=K_RED if bounded else K_BLACK, mew=0.7)
    for K in (2, 4, 8):
        ax.axhline(K + 0.5, color=K_GREY, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.text(0.56, K + 0.5, "$K=" + str(K) + "$", fontsize=6.3, va="center", ha="left",
                color=K_DARK, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(2.5, 1.6, "exact\n$d \\geq k$", fontsize=6.6, ha="center", va="center", color=K_DARK)
    ax.text(1.5, 7.0, "bounded\n$d < k$", fontsize=6.6, ha="center", va="center", color=K_RED)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["2", "4", "8"])
    ax.set_yticks([1, 2, 4, 6, 8])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.3, 8.9)
    ax.set_xlabel("synaptic delay $d$")
    ax.set_ylabel("certificate step $k$")
    panel_label(ax, "(a)", dx=-0.30)

    # ---- (b) R_short against the horizon ----------------------------------------------------
    rshort = np.array([0.000, 0.275, 6.307])
    taps = ["none", "$\\{2\\}$", "$\\{2,4\\}$"]
    floor = 3e-3
    ax = axes[1]
    ax.bar(np.arange(3), np.maximum(rshort, floor), width=0.5, color=K_GREY,
           edgecolor=K_BLACK, lw=0.6, zorder=3)
    ax.axhline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=4)
    ax.text(2.46, BUDGET * 1.35, "budget 0.394", fontsize=6.3, color=K_RED, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 60)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["2", "4", "8"])
    ax.set_xlabel("certificate horizon $K$")
    ax.set_ylabel("$R_{\\mathrm{short}}$")
    for i, r in enumerate(rshort):
        ax.text(i, max(r, floor) * 4.2, taps[i], ha="center", fontsize=6.3, color=K_DARK)
    ax.text(0, floor * 1.25, "$0$", ha="center", va="bottom", fontsize=6.3, color=K_DARK)
    panel_label(ax, "(b)", dx=-0.34)

    # ---- (c) and what that does to certification --------------------------------------------
    pct = np.array([99.2, 94.9, 0.2])
    ax = axes[2]
    ax.plot(np.arange(3), pct, marker="o", color=K_BLUE, mfc="white", mec=K_BLUE, mew=1.0,
            zorder=3)
    for i, p in enumerate(pct):
        ax.annotate(("%.1f" % p) + "%", (i, p), textcoords="offset points",
                    xytext=(0, -12 if i < 2 else 8), ha="center", fontsize=6.3, color=K_BLUE)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["2", "4", "8"])
    ax.set_ylim(-10, 115)
    ax.set_xlabel("certificate horizon $K$")
    ax.set_ylabel("certified (% of oracle)")
    panel_label(ax, "(c)", dx=-0.34)

    save(fig, "fig1_decomposition")


# ---------------------------------------------------------------------------------------------
# Figure 2. R_short governs certifiability; total excitation does not.
#   Manuscript table 1 (tab:governing): six architectures, speaker-disjoint validation, seed 1.
#   Points are numbered; the key lives in the caption so the panels stay readable.
# ---------------------------------------------------------------------------------------------
def figure2():
    # (index, % of oracle, R_short, total R, has certificate penalty)
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

    # hand-placed so no two labels collide
    off_a = {1: (0, -12), 2: (8, -3), 3: (0, 7), 4: (0, -12), 5: (8, -3), 6: (7, 4)}
    ha_a = {1: "center", 2: "left", 3: "center", 4: "center", 5: "left", 6: "left"}
    off_b = {1: (8, -3), 2: (0, -12), 3: (0, 7), 4: (8, -3), 5: (8, -2), 6: (-8, -2)}
    ha_b = {1: "left", 2: "center", 3: "center", 4: "left", 5: "left", 6: "right"}

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"wspace": 0.30})

    for ax, x, xl, off, hal in ((axes[0], rs, "$R_{\\mathrm{short}}$", off_a, ha_a),
                                (axes[1], tot, "total recurrent excitation $R$", off_b, ha_b)):
        for i, r in enumerate(rows):
            ax.plot(x[i], pct[i], marker="o" if cons[i] else "s", ms=5, linestyle="none",
                    mfc=K_BLUE if cons[i] else "white",
                    mec=K_BLUE if cons[i] else K_BLACK, mew=0.8, zorder=3)
            ax.annotate(str(r[0]), (x[i], pct[i]), textcoords="offset points",
                        xytext=off[r[0]], ha=hal[r[0]], fontsize=7, color=K_DARK, zorder=4)
        ax.set_xlabel(xl)
        ax.set_ylim(-14, 118)
        ax.set_ylabel("certified (% of oracle)")

    axes[0].set_xscale("log")
    axes[0].set_xlim(0.19, 11)
    axes[0].axvline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    axes[0].text(0.355, 60, "budget", fontsize=6.4, color=K_RED, ha="right", rotation=90,
                 va="center")
    axes[1].set_xlim(3.2, 17.6)

    leg = [Line2D([], [], marker="s", ls="none", ms=5, mfc="white", mec=K_BLACK, mew=0.8,
                  label="no certificate penalty"),
           Line2D([], [], marker="o", ls="none", ms=5, mfc=K_BLUE, mec=K_BLUE,
                  label="certificate penalty")]
    axes[1].legend(handles=leg, loc="center left", bbox_to_anchor=(0.02, 0.52),
                   handletextpad=0.4, labelspacing=0.3)
    panel_label(axes[0], "(a)", dx=-0.20)
    panel_label(axes[1], "(b)", dx=-0.20)

    save(fig, "fig2_governing")


# ---------------------------------------------------------------------------------------------
# Figure 3. The constraint relocates excitation in time.
#   Manuscript table 3 (tab:mech). The SSC points are the four individual test seeds.
# ---------------------------------------------------------------------------------------------
def figure3():
    shd_c = np.array([4.97, 4.90, 5.40])
    shd_k = np.array([0.28, 6.03, 6.46])
    ssc_c = np.array([5.73, 6.12, 7.19])
    ssc_seeds = np.array([[0.29, 8.56, 9.98],     # seed 1
                          [0.29, 8.54, 9.84],     # seed 2
                          [0.30, 7.65, 8.89],     # seed 3
                          [0.29, 8.21, 9.42]])    # seed 4
    ssc_k = ssc_seeds.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"wspace": 0.26})
    w = 0.34
    x = np.arange(3)

    for ax, ctrl, cons, seeds, head in (
            (axes[0], shd_c, shd_k, None,
             "SHD    total $15.27 \\rightarrow 12.76$  ($-16$%)"),
            (axes[1], ssc_c, ssc_k, ssc_seeds,
             "SSC    total $19.04 \\rightarrow 18.83$  ($-1$%)")):
        ax.bar(x - w / 2, ctrl, w, color="white", edgecolor=K_BLACK, lw=0.7,
               label="control", zorder=3)
        ax.bar(x + w / 2, cons, w, color=K_GREY, edgecolor=K_BLACK, lw=0.7,
               label="constrained", zorder=3)
        if seeds is not None:
            for s in seeds:
                ax.plot(x + w / 2, s, marker="o", ms=2.6, mfc=K_BLACK, mec="none",
                        linestyle="none", zorder=5)
        ax.axhline(BUDGET, color=K_RED, lw=0.8, ls=(0, (4, 2)), zorder=4)
        ax.annotate("", xy=(w / 2, cons[0] + 1.3), xytext=(-w / 2, ctrl[0] + 0.35),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=K_RED,
                                    connectionstyle="arc3,rad=-0.3"), zorder=6)
        ax.text(0.30, ctrl[0] + 1.9, "$-94$%", fontsize=7, color=K_RED, ha="center")
        ax.text(2.46, BUDGET + 0.30, "budget 0.394", fontsize=6.3, color=K_RED, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(["$d=2$", "$d=4$", "$d=8$"])
        ax.set_ylabel("excitatory drive $R^{(d)}$")
        ax.set_ylim(0, 11.6)
        ax.set_title(head, fontsize=7, color=K_DARK, pad=4)

    axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.97), handletextpad=0.5,
                   labelspacing=0.3, handlelength=1.3)
    panel_label(axes[0], "(a)", dx=-0.17, dy=1.11)
    panel_label(axes[1], "(b)", dx=-0.17, dy=1.11)

    save(fig, "fig3_reallocation")


# ---------------------------------------------------------------------------------------------
# Figure 4. Where barrier-free execution pays.
#   (a) manuscript table 4 (tab:speed), shared-memory engine, 80/80 runs bit-identical.
#   (b,c) manuscript table 5 (tab:tcp), two-process TCP engine, 3 repeats, 54/54 bit-identical.
#   Panel (b) is the ratio of the table's medians, so both speed panels share one axis meaning.
# ---------------------------------------------------------------------------------------------
def figure4():
    cores = np.array([4, 8, 16, 32])
    lats = [0, 5, 20, 100, 500]
    sm = {0:   [0.93, 0.90, 0.79, 1.44],
          5:   [0.95, 0.88, 0.79, 1.35],
          20:  [0.91, 0.90, 0.78, 1.30],
          100: [1.07, 1.02, 1.05, 1.25],
          500: [1.11, 1.11, 1.11, 1.49]}
    ctrl32 = [0.99, 0.98, 0.83, 0.98, 0.99]        # unconstrained control, 32 cores

    L = np.array([0, 200, 400, 800, 1600])
    hs = np.array([13.08, 17.22, 26.71, 47.09, 86.50])
    ce = np.array([17.59, 19.73, 26.03, 41.67, 73.44])
    sk = np.array([17.57, 19.71, 28.01, 45.03, 79.28])
    blk = {"handshake": [0.13, 3.70, 13.35, 32.53, 73.15],
           "cert":      [0.10, 1.12, 7.43, 22.85, 54.85],
           "cert+skip": [0.10, 1.94, 8.82, 26.59, 60.92]}

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2), gridspec_kw={"wspace": 0.42})
    styles = {"handshake": (K_BLACK, "s", "-"),
              "cert": (K_BLUE, "o", "-"),
              "cert+skip": (K_RED, "^", (0, (4, 2)))}

    # ---- (a) shared memory, speed-up against core count -------------------------------------
    ax = axes[0]
    shades = ["#000000", "#454545", "#787878", K_BLUE, K_RED]
    marks = ["o", "s", "^", "D", "v"]
    for lat, mk, cl in zip(lats, marks, shades):
        ax.plot(cores, sm[lat], marker=mk, color=cl, mfc="white", mec=cl, mew=0.8,
                label="$L=" + str(lat) + "$", zorder=3)
    ax.axhline(1.0, color=K_GREY, lw=0.7, ls=":", zorder=2)
    ax.plot([32] * len(ctrl32), ctrl32, marker="x", color=K_GREY, mew=1.0, ms=4.5,
            linestyle="none", zorder=4)
    ax.text(30, 0.845, "control", fontsize=6.3, color=K_GREY, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    ax.set_xlabel("cores")
    ax.set_ylabel("speed-up over handshake")
    ax.set_ylim(0.70, 1.62)
    ax.legend(loc="upper left", handletextpad=0.4, labelspacing=0.18, handlelength=1.3,
              fontsize=6.3)
    panel_label(ax, "(a)", dx=-0.34)

    # ---- (b) two processes, speed-up against per-message latency ----------------------------
    ax = axes[1]
    for k, y in (("cert", hs / ce), ("cert+skip", hs / sk)):
        cl, mk, ls = styles[k]
        ax.plot(L, y, marker=mk, color=cl, ls=ls, mfc="white", mec=cl, mew=0.8, label=k,
                zorder=3)
    ax.axhline(1.0, color=K_GREY, lw=0.7, ls=":", zorder=2)
    ax.axvline(348, color=K_GREY, lw=0.7, ls=(0, (1, 2)), zorder=2)
    ax.text(395, 0.76, "break-even\n$348\\,\\mu$s", fontsize=6.3, color=K_DARK, va="bottom")
    ax.set_xlabel("per-message latency $L$ ($\\mu$s)")
    ax.set_ylabel("speed-up over handshake")
    ax.set_xlim(-90, 1740)
    ax.set_ylim(0.70, 1.26)
    ax.legend(loc="lower right", handletextpad=0.4, labelspacing=0.22, handlelength=1.6,
              fontsize=6.5)
    panel_label(ax, "(b)", dx=-0.34)

    # ---- (c) where the time actually goes ---------------------------------------------------
    ax = axes[2]
    for k in ("handshake", "cert", "cert+skip"):
        cl, mk, ls = styles[k]
        ax.plot(L, blk[k], marker=mk, color=cl, ls=ls, mfc="white", mec=cl, mew=0.8,
                label=k, zorder=3)
    ax.set_xlabel("per-message latency $L$ ($\\mu$s)")
    ax.set_ylabel("blocked on peer (ms)")
    ax.set_xlim(-90, 1740)
    ax.set_ylim(-5, 88)
    ax.legend(loc="upper left", handletextpad=0.4, labelspacing=0.22, handlelength=1.6,
              fontsize=6.5)
    panel_label(ax, "(c)", dx=-0.34)

    save(fig, "fig4_execution")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nfigures in " + os.path.relpath(OUT))
