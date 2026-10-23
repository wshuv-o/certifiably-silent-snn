#!/usr/bin/env bash
# SSC-001 data download (train + test), via fetch_ssc in s2_strong.py
source ~/research/venv_gpu/bin/activate
cd "$(dirname "$0")"
python -c "
import os; os.environ['DATASET']='ssc'
import s2_strong as s
for sp in ('test','train'):
    D, y = s.fetch_ssc(sp); print(sp, len(y), 'classes', int(y.max())+1, flush=True)
" 2>&1 | grep -vE "Warning|warn\("
ls -la ~/research/data/ssc; df -h / | tail -1