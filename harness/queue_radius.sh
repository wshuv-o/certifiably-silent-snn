#!/usr/bin/env bash
# Wait for the GPU to free, then run RADIUS-001. Detached so a session restart cannot kill it.
cd "$(dirname "$0")"
while pgrep -f "s4_improve.py" > /dev/null; do sleep 15; done
exec bash run_radius001.sh
