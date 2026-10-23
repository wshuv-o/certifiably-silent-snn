"""HW-004: place and route of the neuron unit and the K-stage certificate unit in SkyWater 130 nm (sky130_fd_sc_hd,
tt 25 C 1.8 V) with OpenROAD. Each combinational datapath is wrapped in input and output registers (decay factors,
threshold and adaptation gain are programmable register inputs), so the routed design yields area, the register-to-
register critical path and power. Flow: Yosys mapping -> floorplan (50 % utilisation) -> tap cells -> global and
detailed placement -> clock tree -> global and detailed routing -> STA and power (20 % input toggle activity).
Usage: python pnr.py <name> <K> <clock_ns>"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from amaranth import *
from amaranth.back import rtlil
from core import CertUnit, F

W = 24
HOME = os.path.expanduser("~")
LIBDIR = f"{HOME}/pdk/sky130A/libs.ref/sky130_fd_sc_hd"
LIB = f"{LIBDIR}/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
YOSYS = f"{HOME}/research/venv_gpu/bin/yowasp-yosys"; OPENROAD = f"{HOME}/eda/bin/openroad"


class RegCU(Elaboratable):
    def __init__(self, K):
        self.K = K
        self.beta = Signal(F + 1); self.rho = Signal(F + 1); self.theta = Signal(signed(W)); self.gamma = Signal(signed(W))
        self.cu = CertUnit(K, W, self.beta, self.theta, self.rho, self.gamma)
        self.i_v = Signal(signed(W)); self.i_a = Signal(signed(W)); self.i_s = Signal(); self.i_bmask = Signal()
        self.i_x = [Signal(signed(W), name=f"i_x{k}") for k in range(K)]
        self.i_b = [Signal(signed(W), name=f"i_b{k}") for k in range(K)]
        self.o_v = Signal(signed(W)); self.o_a = Signal(signed(W)); self.o_s = Signal(); self.o_fire = Signal()
        self.o_vk = Signal(signed(W)); self.o_ak = Signal(signed(W))

    def elaborate(self, platform):
        m = Module(); cu = self.cu; m.submodules.cu = cu
        m.d.sync += [cu.v.eq(self.i_v), cu.a.eq(self.i_a), cu.s.eq(self.i_s), cu.bmask.eq(self.i_bmask)]
        m.d.sync += [cu.x[k].eq(self.i_x[k]) for k in range(self.K)] + [cu.b[k].eq(self.i_b[k]) for k in range(self.K)]
        m.d.sync += [self.o_v.eq(cu.v_new), self.o_a.eq(cu.ak[0]), self.o_s.eq(cu.s_new), self.o_fire.eq(cu.fire.any()),
                     self.o_vk.eq(cu.u[self.K - 2] if self.K > 1 else cu.u[0]),
                     self.o_ak.eq(cu.ak[self.K - 2] if self.K > 1 else cu.ak[0])]
        return m

    def ports(self):
        return [self.beta, self.rho, self.theta, self.gamma, self.i_v, self.i_a, self.i_s, self.i_bmask, *self.i_x,
                *self.i_b, self.o_v, self.o_a, self.o_s, self.o_fire, self.o_vk, self.o_ak]


class PipeCU(Elaboratable):
    """K-stage pipelined certificate unit: one neuron per cycle, latency K + 1. Stage k registers u_k, a_k, the
    running OR of the fire flags and the operands of later stages; stage 1 also yields the neuron update."""

    def __init__(self, K):
        RegCU.__init__(self, K)

    ports = RegCU.ports

    def elaborate(self, platform):
        m = Module(); K = self.K; S = lambda n, sh=signed(W): Signal(sh, name=n)
        v, a, sb, bm = S("p_v"), S("p_a"), Signal(name="p_s"), Signal(name="p_bm")
        x = [S(f"p0_x{k}") for k in range(K)]; b = [S(f"p0_b{k}") for k in range(K)]
        m.d.sync += [v.eq(self.i_v), a.eq(self.i_a), sb.eq(self.i_s), bm.eq(self.i_bmask)]
        m.d.sync += [x[k].eq(self.i_x[k]) for k in range(K)] + [b[k].eq(self.i_b[k]) for k in range(K)]
        u_prev, a_prev, fire = v, a, Const(0)
        vn = sn = a1 = None; uk_tap = ak_tap = None
        for k in range(K):
            bk = Mux(bm, b[k], 0) if k > 0 else 0
            g = Mux(sb, self.gamma, 0) if k == 0 else 0
            uc = S(f"uc{k}"); ac = S(f"ac{k}")
            m.d.comb += [uc.eq(((u_prev * self.beta) >> F) + x[k] + bk), ac.eq(((a_prev * self.rho) >> F) + g)]
            fc = uc >= self.theta + ac
            ur, ar, fr = S(f"u{k}"), S(f"a{k}"), Signal(name=f"f{k}")
            m.d.sync += [ur.eq(uc), ar.eq(ac), fr.eq(fire | fc)]
            if k == 0:
                vn, sn, a1 = S("vn"), Signal(name="sn"), S("a1")
                m.d.sync += [sn.eq(fc), vn.eq(uc - Mux(fc, self.theta + ac, 0)), a1.eq(ac)]
            else:                                              # carry stage-1 results down the pipeline
                vn2, sn2, a12 = S(f"vn{k}"), Signal(name=f"sn{k}"), S(f"a1_{k}")
                m.d.sync += [vn2.eq(vn), sn2.eq(sn), a12.eq(a1)]; vn, sn, a1 = vn2, sn2, a12
            if k == max(K - 2, 0):
                uk_tap, ak_tap = ur, ar
            elif uk_tap is not None:
                u2, a2 = S(f"ukt{k}"), S(f"akt{k}"); m.d.sync += [u2.eq(uk_tap), a2.eq(ak_tap)]; uk_tap, ak_tap = u2, a2
            nx = [S(f"p{k+1}_x{j}") for j in range(K)]; nb = [S(f"p{k+1}_b{j}") for j in range(K)]
            m.d.sync += [nx[j].eq(x[j]) for j in range(k + 1, K)] + [nb[j].eq(b[j]) for j in range(k + 1, K)]
            nbm, nsb = Signal(name=f"bm{k+1}"), Signal(name=f"sb{k+1}"); m.d.sync += [nbm.eq(bm), nsb.eq(sb)]
            x, b, bm, sb = nx, nb, nbm, nsb
            u_prev, a_prev, fire = ur, ar, fr
        m.d.comb += [self.o_v.eq(vn), self.o_a.eq(a1), self.o_s.eq(sn), self.o_fire.eq(fire),
                     self.o_vk.eq(uk_tap), self.o_ak.eq(ak_tap)]
        return m


name, K, clk = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "pnr_" + name); os.makedirs(out, exist_ok=True)
if name.startswith("lcu"):
    from lowcert import LowCU
    top = LowCU()
else:
    top = PipeCU(K) if os.environ.get('PIPE') else RegCU(K)
open(f"{out}/{name}.il", "w").write(rtlil.convert(top, ports=top.ports(), name="top"))
subprocess.run([YOSYS, "-q", "-p",
                f"read_rtlil {out}/{name}.il; hierarchy -top top; proc; flatten; opt; wreduce; alumacc; opt; techmap; "
                f"opt -fast; dfflibmap -liberty {LIB}; abc -D {clk*1000:.0f} -liberty {LIB}; "
                f"hilomap -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO; splitnets; opt_clean -purge; "
                f"setundef -zero; insbuf -buf sky130_fd_sc_hd__buf_2 A X; opt_clean; "
                f"tee -o {out}/synth.stat stat -liberty {LIB}; write_verilog -noattr -noexpr -nohex -nodec {out}/{name}.nl.v"],
               check=True)
nl = open(f"{out}/{name}.nl.v").read().replace(" signed ", " ")      # OpenSTA's reader has no 'signed'
open(f"{out}/{name}.nl.v", "w").write(nl)
open(f"{out}/top.sdc", "w").write(
    f"create_clock -name clk -period {clk} [get_ports clk]\n"
    f"set_input_delay 0 -clock clk [delete_from_list [all_inputs] [get_ports clk]]\n"
    f"set_output_delay 0 -clock clk [all_outputs]\nset_load 0.02 [all_outputs]\n")
tcl = f"""
read_lef {LIBDIR}/techlef/sky130_fd_sc_hd__nom.tlef
read_lef {LIBDIR}/lef/sky130_fd_sc_hd.lef
read_liberty {LIB}
read_verilog {out}/{name}.nl.v
link_design top
read_sdc {out}/top.sdc
set_wire_rc -signal -layer met2
set_wire_rc -clock -layer met5
initialize_floorplan -utilization 50 -aspect_ratio 1 -core_space 5 -site unithd
source {out}/tracks.tcl
place_pins -hor_layers met3 -ver_layers met2
tapcell -distance 14 -tapcell_master sky130_fd_sc_hd__tapvpwrvgnd_1
global_placement -density 0.6
estimate_parasitics -placement
buffer_ports
repair_design
detailed_placement
estimate_parasitics -placement
clock_tree_synthesis -buf_list {{sky130_fd_sc_hd__clkbuf_4 sky130_fd_sc_hd__clkbuf_8}} -root_buf sky130_fd_sc_hd__clkbuf_8
set_propagated_clock [all_clocks]
estimate_parasitics -placement
repair_timing -setup
detailed_placement
check_placement
set_routing_layers -signal met1-met5 -clock met1-met5
global_route -allow_congestion
detailed_route -output_drc {out}/drc.rpt -verbose 0
estimate_parasitics -global_routing
set_power_activity -input -activity 0.2
set_power_activity -pins [all_registers -output_pins] -activity 0.2
puts "AREA_CORE_UM2 [expr [ord::get_db_core] ne {{}} ? 0 : 0]"
report_design_area
report_check_types -max_slew -max_capacitance -max_fanout -violators
report_worst_slack -max
report_tns
report_power
report_checks -path_delay max -format full_clock_expanded -digits 3
write_def {out}/{name}.def
write_verilog {out}/{name}.routed.v
exit
"""
# routing tracks of sky130 (pitch/offset from the OpenLane sky130 configuration)
open(f"{out}/tracks.tcl", "w").write("""make_tracks li1  -x_offset 0.23 -x_pitch 0.46 -y_offset 0.17 -y_pitch 0.34
make_tracks met1 -x_offset 0.17 -x_pitch 0.34 -y_offset 0.17 -y_pitch 0.34
make_tracks met2 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.23 -y_pitch 0.46
make_tracks met3 -x_offset 0.34 -x_pitch 0.68 -y_offset 0.34 -y_pitch 0.68
make_tracks met4 -x_offset 0.46 -x_pitch 0.92 -y_offset 0.46 -y_pitch 0.92
make_tracks met5 -x_offset 1.70 -x_pitch 3.40 -y_offset 1.70 -y_pitch 3.40
""")
tcl = tcl.replace('puts "AREA_CORE_UM2 [expr [ord::get_db_core] ne {} ? 0 : 0]"\n', "")
if os.environ.get("NOBUFPORTS"):
    tcl = tcl.replace("buffer_ports\n", "")
if os.environ.get("NOREPAIR"):                 # small units: FastRoute rejects nets created by repair buffers
    tcl = tcl.replace("buffer_ports\n", "").replace("repair_design\n", "").replace("repair_timing -setup\n", "")
    # fix slew without new nets: upsize every driver of 12 or more sinks to its x4 drive strength
    ups = """set nup 0
foreach inst [get_cells *] {
  set ref [get_property $inst ref_name]
  if {![regexp {^(sky130_fd_sc_hd__[a-z0-9]+)_([0-9]+)$} $ref -> stem sz]} continue
  if {$sz >= 4} continue
  foreach p [get_pins -of_objects $inst -filter "direction == output"] {
    set n [get_nets -of_objects $p]
    if {$n eq ""} continue
    if {[llength [get_pins -of_objects $n]] > 12} {
      if {![catch {replace_cell [get_full_name $inst] ${stem}_4}]} { incr nup }
    }
  }
}
puts "UPSIZED $nup"
"""
    tcl = tcl.replace("detailed_placement\nestimate_parasitics -placement\n", "detailed_placement\nestimate_parasitics -placement\n" + ups + "detailed_placement\n", 1)
open(f"{out}/flow.tcl", "w").write(tcl)
r = subprocess.run([OPENROAD, "-no_init", "-exit", f"{out}/flow.tcl"], capture_output=True, text=True, cwd=out)
open(f"{out}/flow.log", "w").write(r.stdout + r.stderr)
print(r.stdout[-6000:]); print(r.stderr[-2000:])
