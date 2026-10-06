#!/usr/bin/env bash
# Separate GPU environment for S2 (strong models). --no-cache-dir to avoid disk blow-up (filled D: twice before).
set -e
python3 -m venv ~/research/venv_gpu
source ~/research/venv_gpu/bin/activate
pip install -q --no-cache-dir --upgrade pip
pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
pip install -q --no-cache-dir numpy h5py pandas
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(1024,1024,device='cuda'); print('matmul ok', float((x@x).sum())!=0)"
du -sh ~/research/venv_gpu; df -h / | tail -1
