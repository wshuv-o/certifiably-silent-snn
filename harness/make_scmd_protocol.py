"""ACC-011: build ~/research/refs/SCommander-protocol, the published SpikCommander SHD code (Wang et al. 2025,
github.com/JackieWang9811/SCommander, best_config_SHD_former) under our selection protocol.

The reference validates SHD on the test set (SHD_dataloaders returns (train, test) and main passes the test
loader as validation). Here speakers 3 and 6 are held out of training as validation, the best epoch is chosen
on that validation set, and the test accuracy at that epoch is reported. Model, optimiser, schedule, epochs,
augmentation (time/neuron masking, applied per training sample) and input binning are the reference's.
Two substitutions keep it in our environment and change no computation: spikingjelly's torch backend in
place of cupy, and a local copy of spikingjelly.datasets.padded_sequence_mask (whose package import pulls in
torchvision). The repository's utils module is not published; build_optimizer is AdamW(lr_w, weight_decay)
as the config names it. Run inside WSL with the venv_gpu python.
"""
import os, shutil

SRC = os.path.expanduser("~/research/refs/SCommander")
DST = os.path.expanduser("~/research/refs/SCommander-protocol")
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(".git", "checkpoints", "log_files"))

s = open(f"{DST}/spikcommder.py").read()
old = "from spikingjelly.datasets import padded_sequence_mask\n"
assert old in s
s = s.replace(old, "from scmd_protocol_data import padded_sequence_mask\n")
open(f"{DST}/spikcommder.py", "w").write(s)

b = open(f"{DST}/module/spikcommander_backbone.py").read()
for imp in ("import matplotlib.pyplot as plt\n", "import seaborn as sns\n", "import matplotlib\n"):
    b = b.replace(imp, "")                          # plotting imports, never called
b = b.replace("matplotlib.use('Agg')\n", "")
# our spikingjelly build has no `backend` argument; its only backend is torch, the substitution named above
import re as _re
b = _re.sub(r",\s*backend\s*=\s*config\.backend", "", b)
# The published backbone defines MSTASA_v_branch.forward and SCRMLP.forward at module level (column 0), so
# neither class has a forward and the published code cannot run. Re-indent each module-level forward into
# the class it follows; the function bodies are unchanged.
L = b.split("\n"); out = []; inside = False
for line in L:
    if line.startswith("def forward("):
        inside = True
    elif inside and line and not line[0].isspace():
        inside = False                               # next top-level statement (class / def) ends it
    out.append(("    " + line if line.strip() else line) if inside else line)
b = "\n".join(out)
assert "\ndef forward(" not in b
s2 = _re.sub(r",\s*backend\s*=\s*config\.backend", "", open(f"{DST}/spikcommder.py").read())
open(f"{DST}/spikcommder.py", "w").write(s2)
assert "plt." not in b and "sns." not in b
open(f"{DST}/module/spikcommander_backbone.py", "w").write(b)

open(f"{DST}/scmd_protocol_data.py", "w").write('''import os
import numpy as np
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

D = os.path.expanduser("~/research/data/shd")


def padded_sequence_mask(sequence_len, T=None):
    """spikingjelly.datasets.padded_sequence_mask: (T, N) bool, True on valid steps."""
    if T is None:
        T = int(sequence_len.max().item())
    N = sequence_len.numel()
    t_seq = torch.arange(0, T).unsqueeze(1).repeat(1, N).to(sequence_len)
    return t_seq < sequence_len.unsqueeze(0).repeat(T, 1)


def pad_sequence_collate(batch):
    xs = [torch.as_tensor(x) for x, _ in batch]
    return (pad_sequence(xs, batch_first=True), torch.as_tensor([y for _, y in batch]),
            torch.as_tensor([len(x) for x in xs]))


class TimeNeuronsMask:
    """The reference's TimeNeurons_mask_aug, verbatim in effect."""
    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        c = self.c
        if np.random.uniform() < c.TN_mask_aug_proba:
            m = int(c.time_mask_proportion * x.shape[0])
            if x.shape[0] - m > 0:
                i = np.random.randint(0, x.shape[0] - m); x[i:i + m, :] = 0
        if np.random.uniform() < c.TN_mask_aug_proba:
            m = np.random.randint(0, c.neuron_mask_size)
            i = np.random.randint(0, x.shape[1] - c.neuron_mask_size); x[:, i:i + m] = 0
        return x


class SHDFrames(Dataset):
    def __init__(self, split, n_bins, dt_ms, idx=None, aug=None):
        f = h5py.File(f"{D}/shd_{split}.h5", "r")
        self.t = f["spikes"]["times"][:]; self.u = f["spikes"]["units"][:]
        self.y = f["labels"][:].astype(np.int64); self.spk = f["extra"]["speaker"][:]
        self.idx = np.arange(len(self.y)) if idx is None else np.asarray(idx)
        self.n_bins, self.dt, self.aug = n_bins, dt_ms / 1000.0, aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]; t, u = self.t[i], self.u[i]
        nT = max(1, int(np.ceil((t.max() if len(t) else 0) / self.dt)))
        fr = np.zeros((nT, 700), np.float32)
        np.add.at(fr, (np.minimum((t / self.dt).astype(int), nT - 1), u.astype(int)), 1.0)
        fr = fr.reshape(nT, 700 // self.n_bins, self.n_bins).sum(-1)
        if self.aug is not None:
            fr = self.aug(fr)
        return fr, int(self.y[i])


def loaders(c, val_speakers=(3, 6)):
    with h5py.File(f"{D}/shd_train.h5", "r") as f:       # speaker ids only (memory: no full copy)
        spk = f["extra"]["speaker"][:]
    isval = np.isin(spk, val_speakers)
    aug = TimeNeuronsMask(c) if c.use_aug else None
    train = SHDFrames("train", c.n_bins, c.time_step, np.nonzero(~isval)[0], aug=aug)
    valid = SHDFrames("train", c.n_bins, c.time_step, np.nonzero(isval)[0])
    test = SHDFrames("test", c.n_bins, c.time_step)
    kw = dict(collate_fn=pad_sequence_collate, batch_size=c.batch_size, num_workers=2)
    return (DataLoader(train, shuffle=True, **kw), DataLoader(valid, **kw), DataLoader(test, **kw))
''')

open(f"{DST}/scmd_main.py", "w").write('''"""ACC-011 runner: reference training loop, protocol selection. Env: SEED, EPOCHS, TAG, TEST."""
import os, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import lr_scheduler
from spikingjelly.activation_based import functional
from best_config_SHD_former import Config
from spikcommder import SpikCommander
from scmd_protocol_data import loaders, padded_sequence_mask

E = os.environ.get
c = Config()
c.seed = int(E("SEED", "0")); c.epochs = int(E("EPOCHS", str(c.epochs))); c.backend = "torch"
TAG = E("TAG", "a11"); TEST = E("TEST", "1") == "1"
c.n_hidden_neurons = c.n_hidden_neurons_list[0]; c.hidden_dims = c.mlp_ratio * c.n_hidden_neurons
np.random.seed(c.seed); torch.manual_seed(c.seed); torch.cuda.manual_seed_all(c.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = torch.device("cuda")
train_loader, valid_loader, test_loader = loaders(c)
model = SpikCommander(c).to(dev)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
opt = torch.optim.AdamW(model.parameters(), lr=c.lr_w, weight_decay=c.weight_decay)
sched = lr_scheduler.CosineAnnealingLR(opt, T_max=c.t_max)
print(f"ACC-011 {TAG} seed {c.seed} epochs {c.epochs} params {n_params}", flush=True)


def m_sum(out):                      # the reference's loss = 'sum': time-summed softmax
    return torch.softmax(out, 2).sum(0)


@torch.no_grad()
def acc(loader):
    model.eval(); hit = n = 0
    for x, y, l in loader:
        am = padded_sequence_mask(l).transpose(0, 1).to(dev)
        p = m_sum(model(x.float().to(dev), am)).argmax(1).cpu()
        hit += int((p == y).sum()); n += len(y); functional.reset_net(model)
    return hit / n


best_val, best_ep, best_sd, test_curve = -1, -1, None, []
t0 = time.time()
for ep in range(c.epochs):
    model.train()
    for x, y, l in train_loader:
        am = padded_sequence_mask(l).transpose(0, 1).to(dev)
        out = model(x.float().to(dev), am)
        loss = nn.CrossEntropyLoss()(m_sum(out), F.one_hot(y, c.n_outputs).float().to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); functional.reset_net(model)
    sched.step()
    va = acc(valid_loader)
    if va > best_val:
        best_val, best_ep, best_sd = va, ep, copy.deepcopy(model.state_dict())
    te = acc(test_loader) if TEST else float("nan")   # logged for the reference's own rule only
    test_curve.append(te)
    print(f"epoch {ep} val {va*100:.2f} best_val {best_val*100:.2f}@{best_ep} ({(time.time()-t0)/(ep+1):.1f}s/ep)", flush=True)
torch.save(best_sd, os.path.expanduser(f"~/research/models/{TAG}_s{c.seed}_Best_ACC.pt"))
model.load_state_dict(best_sd)
res = dict(tag=TAG, seed=c.seed, epochs=c.epochs, n_params=int(n_params), best_epoch=best_ep, acc_val=best_val,
           acc_test=acc(test_loader) if TEST else None,
           test_best_epoch_rule=float(np.nanmax(test_curve)) if TEST else None,
           final_test=test_curve[-1] if TEST else None, minutes=(time.time() - t0) / 60)
print("RESULT " + json.dumps(res), flush=True)
json.dump(res, open(os.path.expanduser(f"~/research/logs/{TAG}_s{c.seed}.json"), "w"), indent=1)
''')
print("SCommander protocol copy ready at", DST)
