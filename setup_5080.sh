#!/usr/bin/env bash
# One-time setup for the RTX 5080 (Blackwell, sm_120) machine. Linux / WSL2 Ubuntu 24.04.
# Creates the same layout the harness scripts expect: ~/research/{venv_gpu,venv_speech,data,models,build}
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential python3-venv python3-pip git wget
mkdir -p ~/research/{data,models,build}

# GPU environment: PyTorch for CUDA 12.8+ (required for Blackwell / RTX 50xx)
python3 -m venv ~/research/venv_gpu
~/research/venv_gpu/bin/pip install -q --no-cache-dir --upgrade pip
~/research/venv_gpu/bin/pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
~/research/venv_gpu/bin/pip install -q --no-cache-dir numpy h5py pandas scipy matplotlib python-docx

# CPU environment used by the small-model scripts, figures and fixed-point checks
python3 -m venv ~/research/venv_speech
~/research/venv_speech/bin/pip install -q --no-cache-dir --upgrade pip
~/research/venv_speech/bin/pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
~/research/venv_speech/bin/pip install -q --no-cache-dir numpy h5py pandas scipy matplotlib

~/research/venv_gpu/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device='cuda'); print('matmul ok', bool((x@x).isfinite().all()))"
echo "Setup done. Datasets download automatically on first run into ~/research/data/."
