#!/bin/bash
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd $REPO/harness/hw/build
for d in nu pcu2 pcu4; do echo "== $d"; ~/eda/bin/openroad -no_init -exit pnr_$d/pmean.tcl 2>&1 | grep -E "^(Sequential|Combinational|Clock|Total) |ERROR|Error"; done
