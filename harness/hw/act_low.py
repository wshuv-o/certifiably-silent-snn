"""HW-007 power: gate-level simulation (Icarus, sky130 functional models) of the routed reduced-precision
certificate unit on real operands (skewed stage inputs, en = 1), mean toggle density of data nets, then OpenROAD
power with that density applied globally (same method as the other units, see N3_SCALEUP_PLAN HW-005).
Usage: python act_low.py <data_dir> <model> <ncycles>"""
import os, re, subprocess, sys, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lowcert import operands, BU, BD, RD, TH, GM, K, W, WL, PB

ddir, model, NC = sys.argv[1], sys.argv[2], int(sys.argv[3])
HOME = os.path.expanduser("~"); L = f"{HOME}/pdk/sky130A/libs.ref/sky130_fd_sc_hd"
out = os.path.join(HERE, "build", "pnr_lcu4"); nl_path = f"{out}/lcu4.routed.v"
ops = operands(ddir, model, NC)
names = ["bu", "bd", "rd", "th", "gm", "en", "i_v", "i_a", "i_s"] + [f"i_x{k}" for k in range(K)] + [f"i_b{k}" for k in range(K)]
width = dict(bu=PB + 1, bd=PB + 1, rd=PB + 1, th=WL, gm=WL, en=1, i_v=W, i_a=W, i_s=1)
width.update({f"i_x{k}": W for k in range(K)}); width.update({f"i_b{k}": W for k in range(K)})
with open(f"{out}/stim.txt", "w") as f:
    for t in range(len(ops) + K + 1):
        v = dict(bu=BU, bd=BD, rd=RD, th=TH, gm=GM, en=1, i_v=0, i_a=0, i_s=0)
        if t < len(ops): v.update(i_v=ops[t][0], i_a=ops[t][1], i_s=ops[t][2])
        for k in range(K):
            j = t - k - 1
            v[f"i_x{k}"] = ops[j][3][k] if 0 <= j < len(ops) else 0
            v[f"i_b{k}"] = ops[j][4][k] if 0 <= j < len(ops) else 0
        f.write(" ".join(format(v[n] & ((1 << width[n]) - 1), "x") for n in names) + "\n")
hdr = re.search(r"module top\s*\((.*?)\);", open(nl_path).read(), re.S).group(1)
ports = [p.strip() for p in hdr.split(",")]
decl = "".join(f"  reg [{width[n] - 1}:0] {n};\n" for n in names)
fmt = " ".join(["%h"] * len(names))
tbv = ("`timescale 1ns/1ps\nmodule bench;\n  reg clk = 0, rst = 1;\n" + decl + "  wire o_fire;\n"
       f"  top dut({', '.join(f'.{p}({p})' for p in ports)});\n"
       "  integer fd, r;\n  always #12 clk = ~clk;\n  initial begin\n"
       f"    fd = $fopen(\"{out}/stim.txt\", \"r\");\n    $dumpfile(\"{out}/gl.vcd\"); $dumpvars(1, dut);\n"
       "    @(negedge clk); rst = 0;\n    while (!$feof(fd)) begin\n"
       f"      r = $fscanf(fd, \"{fmt}\\n\", {', '.join(names)});\n      @(negedge clk);\n    end\n    $finish;\n  end\nendmodule\n")
open(f"{out}/bench.v", "w").write(tbv)
r = subprocess.run(["iverilog", "-g2012", "-DFUNCTIONAL", "-o", f"{out}/gl.vvp", f"{out}/bench.v", nl_path,
                    f"{L}/verilog/primitives.v", f"{L}/verilog/sky130_fd_sc_hd.v"], capture_output=True, text=True)
if r.returncode: print(r.stderr[-3000:]); sys.exit(1)
subprocess.run(["vvp", "-n", f"{out}/gl.vvp"], capture_output=True, text=True)
ids, tog, last = {}, collections.Counter(), {}
for line in open(f"{out}/gl.vcd"):
    if line.startswith("$var"):
        p = line.split(); ids.setdefault(p[3], []).append((p[4], int(p[2])))
    elif line[:1] in ("0", "1", "x", "z") and len(line.strip()) > 1:
        c, v = line[1:].strip(), line[0]
        if c in last and last[c] != v: tog[c] += 1
        last[c] = v
    elif line[:1] == "b":
        v, c = line[1:].split(); pv = last.get(c)
        if pv is not None:
            w = max(len(v), len(pv)); tog[c] += sum(1 for x, y in zip(v.zfill(w), pv.zfill(w)) if x != y)
        last[c] = v
dens = [tog[c] / len(ops) / w for c, nm in ids.items() for n, w in nm if not n.startswith("clk") and n not in names]
dens = [x for x in dens if x < 1.5]
m = statistics.mean(dens)
head = open(os.path.join(HERE, "build", "pnr_nu", "pmean.tcl")).read().split("read_def")[0]
tcl = (head + f"read_def pnr_lcu4/lcu4.def\nread_sdc pnr_lcu4/top.sdc\nset_wire_rc -signal -layer met2\nset_wire_rc -clock -layer met5\n"
       "set_propagated_clock [all_clocks]\nestimate_parasitics -placement\n"
       f"set_power_activity -global -activity {m:.4f}\nset_power_activity -input -activity {m:.4f}\n"
       "set_power_activity -input_ports clk -activity 2\nreport_power\nexit\n")
open(f"{out}/pmean.tcl", "w").write(tcl)
r = subprocess.run([f"{HOME}/eda/bin/openroad", "-no_init", "-exit", f"{out}/pmean.tcl"], capture_output=True, text=True,
                   cwd=os.path.join(HERE, "build"))
tot = [l for l in r.stdout.splitlines() if l.startswith("Total ")]
print(f"RESULT lcu4 model={model} cycles={len(ops)} mean_density={m:.4f} nets={len(dens)} {tot[-1] if tot else r.stdout[-800:]}")
