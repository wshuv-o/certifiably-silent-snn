#!/usr/bin/env bash
# Render the NCE manuscript with Tectonic (installed without sudo at ~/.local/bin/tectonic).
# Tectonic fetches the TeX Live bundle on first run, including iopart.cls if it is present there.
cd /mnt/d/proc/certifiably-silent-snn/research/paper
echo "=== tectonic $(~/.local/bin/tectonic --version) ==="
~/.local/bin/tectonic -X compile manuscript_nce.tex --keep-logs
rc=$?
echo "=== tectonic exit=$rc ==="
ls -la manuscript_nce.pdf manuscript_nce.log 2>&1 | tail -4
[ -f manuscript_nce.log ] && { echo "=== log errors ==="; grep -iE "^!|error|not found|undefined" manuscript_nce.log | head -20; }
