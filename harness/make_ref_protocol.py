"""Create ~/research/refs/SNN-delays-protocol: the published DCLS code (Hammouamri et al. 2024) with ONE change,
the selection protocol. The reference validates SHD on the test set and reports the best test epoch; here
speakers 3 and 6 are held out of training as validation, the best epoch is chosen on that validation set,
and the test accuracy at that epoch is reported. Model, optimiser, schedules and delay learning are untouched.
Run inside WSL with the venv_gpu python."""
import os, shutil, re

SRC = os.path.expanduser("~/research/refs/SNN-delays")
DST = os.path.expanduser("~/research/refs/SNN-delays-protocol")
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(".git"))

open(os.path.join(DST, "protocol_data.py"), "w").write('''"""Speaker-disjoint SHD loaders producing the reference's input: 10 ms frames of spike counts over 700
channels, adjacent channels summed in groups of n_bins (700 -> 140), variable length, zero-padded per batch."""
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def pad_sequence_collate(batch):
    # same output as spikingjelly.datasets.pad_sequence_collate: (padded [B,T,F], labels, lengths)
    xs = [torch.as_tensor(x) for x, _ in batch]
    return (pad_sequence(xs, batch_first=True), torch.as_tensor([y for _, y in batch]),
            torch.as_tensor([len(x) for x in xs]))

D = os.path.expanduser("~/research/data/shd")


class SHDFrames(Dataset):
    def __init__(self, split, n_bins, dt_ms, idx=None):
        f = h5py.File(f"{D}/shd_{split}.h5", "r")
        self.t = f["spikes"]["times"][:]; self.u = f["spikes"]["units"][:]
        self.y = f["labels"][:].astype(np.int64); self.spk = f["extra"]["speaker"][:]
        self.idx = np.arange(len(self.y)) if idx is None else np.asarray(idx)
        self.n_bins, self.dt = n_bins, dt_ms / 1000.0

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]; t, u = self.t[i], self.u[i]
        nT = max(1, int(np.ceil((t.max() if len(t) else 0) / self.dt)))
        fr = np.zeros((nT, 700), np.float32)
        np.add.at(fr, (np.minimum((t / self.dt).astype(int), nT - 1), u.astype(int)), 1.0)
        fr = fr.reshape(nT, 700 // self.n_bins, self.n_bins).sum(-1)
        return fr, int(self.y[i])


def loaders(config, val_speakers=(3, 6)):
    tr = SHDFrames("train", config.n_bins, config.time_step)
    isval = np.isin(tr.spk, val_speakers)
    train = SHDFrames("train", config.n_bins, config.time_step, np.nonzero(~isval)[0])
    valid = SHDFrames("train", config.n_bins, config.time_step, np.nonzero(isval)[0])
    test = SHDFrames("test", config.n_bins, config.time_step)
    kw = dict(collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)
    return (DataLoader(train, shuffle=True, **kw), DataLoader(valid, **kw), DataLoader(test, **kw))
''')

# main.py: use the speaker-disjoint loaders for SHD
m = open(os.path.join(DST, "main.py")).read()
old = "    train_loader, valid_loader = SHD_dataloaders(config)\n    test_loader = None"
assert old in m
m = m.replace(old, "    from protocol_data import loaders\n    train_loader, valid_loader, test_loader = loaders(config)")
open(os.path.join(DST, "main.py"), "w").write(m)

# config.py: seed and save path from the environment
c = open(os.path.join(DST, "config.py")).read()
c = re.sub(r"\n    seed = 0\n", "\n    seed = int(__import__('os').environ.get('SEED', '0'))\n", c, count=1)
assert "environ.get('SEED'" in c
c = c.replace("    save_model_path = f'{wandb_run_name}_REPL.pt'",
              "    save_model_path = __import__('os').path.expanduser(f\"~/research/models/ref_s{seed}_REPL.pt\")")
assert "ref_s{seed}" in c
open(os.path.join(DST, "config.py"), "w").write(c)

# model.py: remember the test accuracy at the best-validation epoch, report it at the end
md = open(os.path.join(DST, "model.py")).read()
old = "                best_metric_val = metric_valid\n"
assert md.count(old) == 1
md = md.replace(old, old + "                self._test_at_best_val = metric_test; self._best_epoch = epoch\n")
old2 = "            if  loss_valid < best_loss_val:"
assert md.count(old2) == 1
md = md.replace(old2, "            if epoch == self.config.epochs - 1:\n"
                "                import json\n"
                "                print('PROTOCOL_RESULT ' + json.dumps(dict(seed=self.config.seed, best_epoch=int(self._best_epoch), "
                "best_val=float(best_metric_val), test_at_best_val=float(self._test_at_best_val), "
                "final_val=float(metric_valid), final_test=float(metric_test))), flush=True)\n" + old2)
open(os.path.join(DST, "model.py"), "w").write(md)
print("protocol copy ready at", DST)

# The SHD path needs none of torchvision / torchaudio (used only by the GSC/SSC loaders and the
# augmentations, which best_config_SHD switches off); drop those imports so the copy runs in our venv.
m = open(os.path.join(DST, "main.py")).read()
m = m.replace("from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders\n", "")
open(os.path.join(DST, "main.py"), "w").write(m)
md = open(os.path.join(DST, "model.py")).read()
assert "from datasets import Augs\n" in md
md = md.replace("from datasets import Augs\n", "try:\n    from datasets import Augs\nexcept ImportError:\n"
                "    class Augs:  # constructed unconditionally, applied only when config.augment\n"
                "        def __init__(self, config): pass\n"
                "        def __call__(self, x, y): raise RuntimeError('augment needs the full datasets module')\n")
open(os.path.join(DST, "model.py"), "w").write(md)
print("SHD-only imports patched")
# wandb is used only under config.use_wandb (False in best_config_SHD); stub the import.
md = open(os.path.join(DST, "model.py")).read()
md = md.replace("import wandb\n", "try:\n    import wandb\nexcept ImportError:\n    wandb = None\n", 1)
open(os.path.join(DST, "model.py"), "w").write(md)
print("wandb stubbed")

# ACC-007: optional lower bound on the epoch at which the best-validation checkpoint may be taken, so a
# certificate fine-tune is selected like its control (best validation) but only once lambda is at full
# strength. Default 0 = the reference's behaviour.
md = open(os.path.join(DST, "model.py")).read()
old = "            if  metric_valid > best_metric_val:"
assert md.count(old) == 1
md = md.replace(old, "            if  metric_valid > best_metric_val and epoch >= getattr(self.config, 'select_from_epoch', 0):")
open(os.path.join(DST, "model.py"), "w").write(md)
print("select_from_epoch added")
