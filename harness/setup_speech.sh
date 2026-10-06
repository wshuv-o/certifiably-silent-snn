#!/usr/bin/env bash
# Speech pilot environment: separate, CPU-only venv (~1 GB).
# History: a CUDA PyTorch install (~6 GB of NVIDIA libs) filled drive D: twice on
# 2026-10-04 and corrupted files written at that moment. The pilot (activation-change
# measurement) does not need a GPU; GPU timing comes later if the pilot passes.
set -e
# 1) remove the broken CUDA torch stack from the harness venv (frees ~6 GB inside the vhdx)
source ~/research/venv/bin/activate
pip uninstall -y -q torch transformers tokenizers safetensors huggingface-hub datasets jinja2 \
    $(pip list 2>/dev/null | awk '/^nvidia-/{print $1}') > /dev/null 2>&1 || true
python -c "import numba, cupy; print('harness venv OK: numba', numba.__version__, 'cupy', cupy.__version__)"
deactivate
rm -rf ~/.cache/pip
# 2) fresh CPU-only speech venv
python3 -m venv ~/research/venv_speech
source ~/research/venv_speech/bin/activate
pip install -q --no-cache-dir --upgrade pip
pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install -q --no-cache-dir transformers datasets soundfile librosa jiwer numpy pandas
python -c "import torch, transformers; print('speech venv OK: torch', torch.__version__, '| transformers', transformers.__version__)"
du -sh ~/research/venv ~/research/venv_speech
df -h / | tail -1
