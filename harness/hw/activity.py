"""HW-005: data-driven switching activity for the power analysis.
1. Operand stream: the per-neuron operands each unit sees in a normal pass (root state v, a, s; accumulator slots
   x_1..x_K with host input; bounded drive b_1..b_K), taken from the golden model of a network in core order.
2. Gate-level simulation of the routed netlist (Icarus Verilog, sky130 functional cell models, zero delay).
3. Toggle density per net (toggles per clock) -> `set_power_activity -pins` on every cell output pin, then
   OpenROAD `report_power` on the routed design. Zero-delay simulation omits glitches.
Usage: python activity.py <unit nu|pcu2|pcu4> <K> <data_dir> <model> <ncycles>"""
import os, re, subprocess, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
unit, K, ddir, model, NC = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
from golden import Net
from energy_estimate import load

HOME = os.path.expanduser("~"); L = f"{HOME}/pdk/sky130A/libs.ref/sky130_fd_sc_hd"
out = os.path.join(HERE, "build", f"pnr_{unit}")
BETA, THETA, RHO, GAMMA = 2484, 4096, 3819, 82

# 1. operand stream from the golden model (32-neuron cores, neurons in core order, steps in time order)
net = load(ddir, model)
g = Net(net["W"], net["D"], net["I"], net["beta"], net["theta"], net["rho"], net["gamma"], 32)
Bk = [np.broadcast_to(g.bounded(k + 1), (g.H,)) for k in range(K)]
ops = []
for b in range(net["N"]):
    S, cert, viol, V, A, slots = g.run(b, K=max(K, 2))
    for t in range(1, S.shape[0] - max(K, 2)):
        for i in range(g.H):
            x = [int(slots[t, k, i] + g.I[b, t + 1 + k, i]) for k in range(K)]
            ops.append((int(V[t, i]), int(A[t, i]), int(S[t, i]), 1, x, [int(Bk[k][i]) for k in range(K)]))
            if len(ops) >= NC: break
        if len(ops) >= NC: break
    if len(ops) >= NC: break

# 2. gate-level simulation
nl_path = f"{out}/{unit}.routed.v"
hdr = re.search(r"module top\s*\((.*?)\);", open(nl_path).read(), re.S).group(1)
names_in = ["beta", "rho", "theta", "gamma", "i_v", "i_a", "i_s", "i_bmask"] + [f"i_x{k}" for k in range(K)] + [f"i_b{k}" for k in range(K)]
width = dict(beta=13, rho=13, theta=24, gamma=24, i_v=24, i_a=24, i_s=1, i_bmask=1)
width.update({f"i_x{k}": 24 for k in range(K)}); width.update({f"i_b{k}": 24 for k in range(K)})
with open(f"{out}/stim.txt", "w") as f:
    for v, a, sb, bm, x, bb in ops:
        vals = dict(beta=BETA, rho=RHO, theta=THETA, gamma=GAMMA, i_v=v, i_a=a, i_s=sb, i_bmask=bm)
        vals.update({f"i_x{k}": x[k] for k in range(K)}); vals.update({f"i_b{k}": bb[k] for k in range(K)})
        f.write(" ".join(format(vals[n] & ((1 << width[n]) - 1), "x") for n in names_in) + "\n")
ports = [p.strip() for p in hdr.split(",")]
decl = "".join(f"  reg [{width[n] - 1}:0] {n};\n" for n in names_in)
conn = ", ".join(f".{p}({p})" for p in ports)
outs = [p for p in ports if p not in names_in and p not in ("clk", "rst")]
fmt = " ".join(["%h"] * len(names_in))
tbv = (
    "`timescale 1ns/1ps\n"
    "module bench;\n"
    "  reg clk = 0, rst = 1;\n"
    f"{decl}"
    f"  wire [63:0] {', '.join(outs)};\n"
    f"  top dut({conn});\n"
    "  integer fd, r, n;\n"
    "  always #12 clk = ~clk;\n"
    "  initial begin\n"
    f"    fd = $fopen(\"{out}/stim.txt\", \"r\");\n"
    f"    $dumpfile(\"{out}/gl.vcd\"); $dumpvars(1, dut);\n"
    "    @(negedge clk); rst = 0; n = 0;\n"
    "    while (!$feof(fd)) begin\n"
    f"      r = $fscanf(fd, \"{fmt}\\n\", {', '.join(names_in)});\n"
    "      @(negedge clk); n = n + 1;\n"
    "    end\n"
    f"    repeat ({K + 2}) @(negedge clk);\n"
    "    $display(\"CYCLES %0d\", n); $finish;\n"
    "  end\n"
    "endmodule\n")
open(f"{out}/bench.v", "w").write(tbv)
r = subprocess.run(["iverilog", "-g2012", "-DFUNCTIONAL", "-o", f"{out}/gl.vvp", f"{out}/bench.v", nl_path,
                    f"{L}/verilog/primitives.v", f"{L}/verilog/sky130_fd_sc_hd.v"], capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
r = subprocess.run(["vvp", "-n", f"{out}/gl.vvp"], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", r.stderr[-500:])
vgl = f"{out}/gl.vcd"

# 3. toggle density per net
ids, toggles, last = {}, collections.Counter(), {}
with open(vgl) as f:
    for line in f:
        if line.startswith("$var"):
            p = line.split(); ids.setdefault(p[3], []).append((p[4], int(p[2])))
        elif line[:1] in ("0", "1", "x", "z") and len(line.strip()) > 1:
            c, v = line[1:].strip(), line[0]
            if c in last and last[c] != v: toggles[c] += 1
            last[c] = v
        elif line[:1] == "b":
            v, c = line[1:].split()
            pv = last.get(c)
            if pv is not None:
                w = max(len(v), len(pv)); a, b2 = v.zfill(w), pv.zfill(w)
                toggles[c] += sum(1 for x, y in zip(a, b2) if x != y)
            last[c] = v
dens = {}
for c, names in ids.items():
    for n, w in names:
        dens[n] = toggles[c] / max(len(ops), 1) / w
print("nets", len(dens), "mean density %.4f" % (np.mean(list(dens.values())) if dens else 0))

# per-pin activity on every cell output, then power on the routed design
nl = open(nl_path).read()
pins, missing = [], 0
for inst, body in re.findall(r"sky130_fd_sc_hd__\w+\s+(\S+)\s*\((.*?)\);", nl, re.S):
    for pin, netn in re.findall(r"\.(X|Y|Q|Q_N|COUT|SUM)\(([^)]*)\)", body):
        key = netn.strip().replace(" ", "")
        dv = dens.get(key, dens.get(key.lstrip("\\")))
        if dv is None:
            key2 = re.sub(r"\[\d+\]$", "", key.lstrip("\\")); dv = dens.get(key2)
        if dv is None: missing += 1; continue
        pins.append((inst.lstrip("\\"), pin, dv))
inp = np.mean([dens[n] for n in names_in if n in dens])
tcl = open(f"{out}/power.tcl").read().split("set_power_activity -input")[0]
tcl += f"set_power_activity -input -activity {inp:.4f}\n"
tcl += "".join(f"set_power_activity -pins {{{i}/{p}}} -activity {d:.5f}\n" for i, p, d in pins)
tcl += "report_power\nexit\n"
open(f"{out}/power_data.tcl", "w").write(tcl)
r = subprocess.run([f"{HOME}/eda/bin/openroad", "-no_init", "-exit", f"{out}/power_data.tcl"], capture_output=True, text=True,
                   cwd=os.path.join(HERE, "build"))
open(f"{out}/power_data.log", "w").write(r.stdout + r.stderr)
tot = [l for l in r.stdout.splitlines() if l.startswith("Total ")]
nwarn = r.stdout.count("WARNING")
print(f"RESULT unit={unit} K={K} model={model} cycles={len(ops)} annotated_pins={len(pins)} missing_pins={missing} "
      f"input_density={inp:.4f} warnings={nwarn} {tot[-1] if tot else 'NO POWER'}")
