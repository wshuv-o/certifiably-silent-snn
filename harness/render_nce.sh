#!/usr/bin/env bash
# Render the NCE manuscript with the real iopart.cls (vendored from rstudio/rticles).
cd /mnt/d/proc/certifiably-silent-snn/research/paper
~/.local/bin/tectonic -X compile manuscript_nce.tex --keep-logs > tec_out.txt 2>&1
echo "=== exit=$? ==="
grep -nE "^error|^!|Undefined|not found" tec_out.txt | head -20
echo "--- log ---"
grep -nE "^!|^l\.[0-9]+|Undefined" manuscript_nce.log 2>/dev/null | head -20
ls -la manuscript_nce.pdf 2>/dev/null
