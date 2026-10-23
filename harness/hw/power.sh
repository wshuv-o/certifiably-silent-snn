#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
# HW-004 power: reload each routed DEF, placement parasitics, activity ACT on inputs and on every flip-flop output
# (same assumption for all units, no compounding through pipeline registers), report power at the run's clock.
cd $REPO/harness/hw/build
L=~/pdk/sky130A/libs.ref/sky130_fd_sc_hd
ACT=${ACT:-0.2}
for d in nu pcu2 pcu4; do
cat > pnr_$d/power.tcl <<TCL
read_lef $L/techlef/sky130_fd_sc_hd__nom.tlef
read_lef $L/lef/sky130_fd_sc_hd.lef
read_liberty $L/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_def pnr_$d/$d.def
read_sdc pnr_$d/top.sdc
set_wire_rc -signal -layer met2
set_wire_rc -clock -layer met5
set_propagated_clock [all_clocks]
estimate_parasitics -placement
set_power_activity -input -activity $ACT
set n 0
foreach p [get_pins -hierarchical */Q] { set_power_activity -pins [get_full_name \$p] -activity $ACT; incr n }
puts "QPINS \$n"
report_power
exit
TCL
~/eda/bin/openroad -no_init -exit pnr_$d/power.tcl > pnr_$d/power.log 2>&1
echo "RESULT $d act=$ACT $(grep QPINS pnr_$d/power.log) $(grep -E '^Total ' pnr_$d/power.log | tail -1) warnings=$(grep -c WARNING pnr_$d/power.log)"
done
