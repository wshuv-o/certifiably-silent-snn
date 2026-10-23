"""Figure: (a) core architecture with the certificate unit, (b) pass schedule of one core, (c) routed sky130 layouts
of the neuron unit and the reduced-precision certificate unit at a common scale.
Usage: python fig_hw.py <build_dir> <lef> <out.pdf>"""
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection, LineCollection

bdir, lef, out = sys.argv[1:4]
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.linewidth": 0.6})
INK, MUTED = "#222222", "#666666"
C_MEM, C_NU, C_CU, C_CTL, C_SYN = "#dbe7f3", "#f6d9b8", "#cfe8d2", "#ececec", "#e7dcef"

fig = plt.figure(figsize=(7.1, 5.0))
axA = fig.add_axes([0.01, 0.43, 0.60, 0.55]); axB = fig.add_axes([0.64, 0.56, 0.35, 0.36])
axC1 = fig.add_axes([0.06, 0.08, 0.30, 0.31]); axC2 = fig.add_axes([0.42, 0.08, 0.30, 0.31])


def box(ax, x, y, w, h, text, fc, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012", fc=fc, ec=INK, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, text=None, dx=0.0, dy=0.012):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6))
    if text: ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=5.5, color=MUTED)


# (a) architecture
ax = axA; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.98, "a", fontsize=9, fontweight="bold", va="top")
box(ax, 0.02, 0.70, 0.20, 0.14, "spike events\n(pre index $j$)", C_CTL)
box(ax, 0.30, 0.66, 0.24, 0.22, "synapse unit\n3 weight tables\n($d$ = 2, 4, 8)", C_SYN)
box(ax, 0.62, 0.62, 0.36, 0.30, "", C_MEM)
ax.text(0.80, 0.885, "delay accumulators", ha="center", va="center", fontsize=6.5)
for b in range(4):
    box(ax, 0.64 + b * 0.085, 0.65, 0.075, 0.17, f"bank\n{b}", "white", fs=5.5)
ax.text(0.80, 0.605, "16-step ring, slot mod 4", ha="center", va="top", fontsize=5.5, color=MUTED)
arrow(ax, 0.22, 0.77, 0.30, 0.77)
arrow(ax, 0.54, 0.77, 0.62, 0.77)
ax.text(0.58, 0.785, "$+w$", ha="center", va="bottom", fontsize=5.5, color=MUTED)
box(ax, 0.02, 0.30, 0.20, 0.20, "neuron state\n$v, a, s$\n(32 neurons)", C_MEM)
box(ax, 0.30, 0.30, 0.22, 0.20, "neuron unit\n(stage 1)\nexact Q12", C_NU)
box(ax, 0.60, 0.30, 0.38, 0.20, "certificate unit, stages 1..$K$\n16-bit, rounded up\nearly abort", C_CU)
arrow(ax, 0.22, 0.44, 0.30, 0.44)
ax.text(0.26, 0.455, "read", ha="center", va="bottom", fontsize=5.5, color=MUTED)
arrow(ax, 0.80, 0.62, 0.80, 0.50, "$x_1..x_K$", dx=0.05, dy=-0.01)
arrow(ax, 0.66, 0.62, 0.41, 0.50, "$x_1$", dx=-0.03, dy=-0.02)
ax.annotate("", xy=(0.60, 0.40), xytext=(0.52, 0.40), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6))
ax.annotate("", xy=(0.22, 0.35), xytext=(0.30, 0.35), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6))
ax.text(0.26, 0.335, "write", ha="center", va="top", fontsize=5.5, color=MUTED)
box(ax, 0.60, 0.04, 0.38, 0.14, "pass controller\nnormal / idle / catch-up", C_CTL)
arrow(ax, 0.79, 0.30, 0.79, 0.18, "core silent\nfor $K$ steps", dx=0.075, dy=-0.03)
box(ax, 0.30, 0.04, 0.22, 0.14, "spike out", C_CTL)
arrow(ax, 0.41, 0.30, 0.41, 0.18)

# (b) pass schedule
ax = axB
kinds = ["N", "N", "N", "I", "I", "C", "N", "N", "I", "I", "C", "N"]
col = {"N": C_NU, "I": "white", "C": C_CU}
for t, k in enumerate(kinds):
    ax.add_patch(Rectangle((t, 0.15), 0.9, 0.7, fc=col[k], ec=INK, lw=0.5))
    ax.text(t + 0.45, 0.5, k, ha="center", va="center", fontsize=6)
for t in (2, 7):
    ax.annotate("", xy=(t + 3.45, 1.05), xytext=(t + 0.45, 1.05), arrowprops=dict(arrowstyle="-|>", lw=0.5, color=MUTED, mutation_scale=5))
    ax.text(t + 1.95, 1.12, "certified", ha="center", va="bottom", fontsize=5.5, color=MUTED)
ax.set_xlim(-0.2, len(kinds)); ax.set_ylim(0, 1.6); ax.set_yticks([])
ax.set_xticks([t + 0.45 for t in range(len(kinds))]); ax.set_xticklabels([str(t + 1) for t in range(len(kinds))], fontsize=5.5)
ax.set_xlabel("time step", fontsize=6)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.text(-0.2, 1.75, "b", fontsize=9, fontweight="bold", va="bottom")
ax.text(len(kinds) / 2, -0.55, "N normal   I idle   C catch-up", ha="center", fontsize=5.5, color=MUTED)

# (c) layouts
size, cur = {}, None
for line in open(lef):
    mm = re.match(r"MACRO (\S+)", line)
    if mm: cur = mm.group(1)
    mm = re.match(r"\s+SIZE ([\d.]+) BY ([\d.]+)", line)
    if mm and cur: size[cur] = (float(mm.group(1)), float(mm.group(2)))


def kind(mst):
    n = mst.replace("sky130_fd_sc_hd__", "")
    if n.startswith(("df", "edf", "sdf")): return "flip-flop"
    if n.startswith(("fa_", "ha_", "maj", "xor", "xnor")): return "adder / XOR"
    if n.startswith(("clkbuf", "clkinv")): return "clock tree"
    if n.startswith(("tap", "fill", "decap", "conb")): return None
    return "other logic"


cc = {"flip-flop": "#1f6fb2", "adder / XOR": "#d9822b", "other logic": "#8fb8de", "clock tree": "#2e9e44"}


def draw(ax, dfile, title, lim):
    d = open(dfile).read(); dbu = float(re.search(r"UNITS DISTANCE MICRONS (\d+)", d).group(1))
    die = [float(v) / dbu for v in re.search(r"DIEAREA \( (-?\d+) (-?\d+) \) \( (-?\d+) (-?\d+) \)", d).groups()]
    comp = d[d.index("COMPONENTS"):d.index("END COMPONENTS")]
    pats = {k: [] for k in cc}
    for mst, x, y, o in re.findall(r"- \S+ (\S+)(?: \+ SOURCE \S+)?\s+\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)", comp):
        k = kind(mst)
        if k:
            w, h = size.get(mst, (0.46, 2.72)); pats[k].append(Rectangle((float(x) / dbu, float(y) / dbu), w, h))
    segs = []
    nets = d[d.index("\nNETS"):d.index("END NETS")]
    for lay, x1, y1, x2, y2 in re.findall(r"(?:ROUTED|NEW) (met[2-5]) \( (-?\d+) (-?\d+)(?: -?\d+)? \) \( (-?\d+|\*) (-?\d+|\*)(?: -?\d+)? \)", nets):
        X1, Y1 = float(x1) / dbu, float(y1) / dbu
        segs.append([(X1, Y1), (X1 if x2 == "*" else float(x2) / dbu, Y1 if y2 == "*" else float(y2) / dbu)])
    for k, ps in pats.items():
        if ps: ax.add_collection(PatchCollection(ps, facecolor=cc[k], edgecolor="none"))
    ax.add_collection(LineCollection(segs, colors="#7b3fa0", linewidths=0.08, alpha=0.35))
    ax.add_patch(Rectangle((die[0], die[1]), die[2] - die[0], die[3] - die[1], fill=False, ec=INK, lw=0.6))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_title(title, fontsize=6.5); ax.tick_params(labelsize=5.5)
    return die


# (c) mask layout of the routed 16-bit certificate unit (KLayout render of DEF + sky130 cell GDS), (d) close-up
import matplotlib.image as mpimg
die_img = mpimg.imread(f"{bdir}/gds_lcu4_die.png"); zoom_img = mpimg.imread(f"{bdir}/gds_lcu4_zoom.png")
D0, D1 = -2.0, 317.605 + 2.0                        # die render extent (um)
Z0, Z1 = 146.3025, 171.3025                         # close-up extent (um)
axC1.imshow(die_img, extent=(D0, D1, D0, D1)); axC2.imshow(zoom_img, extent=(Z0, Z1, Z0, Z1))
axC1.add_patch(Rectangle((Z0, Z0), Z1 - Z0, Z1 - Z0, fill=False, ec="#000000", lw=1.0))
for a, (lo, hi), bar, lab in ((axC1, (D0, D1), 100, "100 $\\mu$m"), (axC2, (Z0, Z1), 5, "5 $\\mu$m")):
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.6)
    x0 = lo + 0.06 * (hi - lo); y0 = lo + 0.05 * (hi - lo)
    a.add_patch(Rectangle((x0 - 0.02 * (hi - lo), y0 - 0.03 * (hi - lo)), bar + 0.04 * (hi - lo), 0.10 * (hi - lo),
                          fc="white", ec="none", alpha=0.85))
    a.plot([x0, x0 + bar], [y0, y0], color="black", lw=1.6)
    a.text(x0 + bar / 2, y0 + 0.015 * (hi - lo), lab, ha="center", va="bottom", fontsize=6)
axC1.set_title("16-bit certificate unit, SkyWater 130 nm", fontsize=6.5)
axC2.set_title("close-up (box in c)", fontsize=6.5)
axC1.text(D0 - 0.08 * (D1 - D0), D1 + 0.06 * (D1 - D0), "c", fontsize=9, fontweight="bold", va="bottom")
axC2.text(Z0 - 0.08 * (Z1 - Z0), Z1 + 0.06 * (Z1 - Z0), "d", fontsize=9, fontweight="bold", va="bottom")
lg = [("diffusion", "#7fbf7f"), ("polysilicon gate", "#d9534f"), ("contact", "#222222"), ("local interconnect", "#a47cc4"),
      ("metal 1", "#4a7fd6"), ("metal 2", "#e08a2c"), ("metal 3", "#2fa36b")]
fig.legend([Rectangle((0, 0), 1, 1, color=c) for _, c in lg], [n for n, _ in lg], loc="lower left",
           bbox_to_anchor=(0.755, 0.12), fontsize=5.8, frameon=False, ncol=1)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("saved", out)
