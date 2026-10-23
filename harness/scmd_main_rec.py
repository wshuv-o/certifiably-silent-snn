"""ACC-012 runner: SpikCommander + certified recurrent layer, our protocol.

Env: SEED, TAG, TEST, EPOCHS. Control (CERT_LAMBDA=0): the reference recipe (500 epochs, AdamW lr 0.01 wd 0.01,
cosine T_max 40, masking augmentation), best epoch on the speaker-disjoint validation set. Fine-tune
(CERT_LAMBDA>0, INIT=control checkpoint): AdamW lr FT_LR (1e-3) wd 0.01, cosine over EPOCHS, lambda ramped
over the first half, best validation epoch among epochs >= SELECT_FROM (lambda at full strength).
The test set is evaluated per epoch only for the reference's own reporting rule; selection never reads it.
"""
import os, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import lr_scheduler
from spikingjelly.activation_based import functional
from best_config_SHD_former import Config
from scmd_protocol_data import loaders, padded_sequence_mask
import scmd_rec as R

E = os.environ.get
c = Config()
c.seed = int(E("SEED", "0")); c.backend = "torch"
TAG = E("TAG", "a12"); TEST = E("TEST", "0") == "1"; INIT = E("INIT", "")
LAM = float(E("CERT_LAMBDA", "0")); SELECT_FROM = int(E("SELECT_FROM", "0"))
c.epochs = int(E("EPOCHS", "100" if INIT else str(c.epochs)))
c.n_hidden_neurons = c.n_hidden_neurons_list[0]; c.hidden_dims = c.mlp_ratio * c.n_hidden_neurons
np.random.seed(c.seed); torch.manual_seed(c.seed); torch.cuda.manual_seed_all(c.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = torch.device("cuda")
train_loader, valid_loader, test_loader = loaders(c)
model = R.SpikCommanderRec(c).to(dev)
if INIT:
    model.load_state_dict(torch.load(INIT, map_location=dev), strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(E("FT_LR", "1e-3")), weight_decay=c.weight_decay)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=c.epochs)
else:
    # ACC-012b: the recurrent layer at REC_LR (default: the host's lr_w). At the host's 0.01 its excitation
    # diverges within a few steps (R(d=2) 2.7 -> 9.8, rate saturating at 0.38) and the network stays at chance.
    rp = list(model.rec.parameters()); ids = {id(p) for p in rp}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in ids], "lr": c.lr_w},
                             {"params": rp, "lr": float(E("REC_LR", str(c.lr_w)))}], weight_decay=c.weight_decay)
    sched = lr_scheduler.CosineAnnealingLR(opt, T_max=c.t_max)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ACC-012 {TAG} seed {c.seed} epochs {c.epochs} lambda {LAM} init {bool(INIT)} params {n_params}", flush=True)


def m_sum(out):
    return torch.softmax(out, 2).sum(0)


@torch.no_grad()
def acc(loader):
    model.eval(); hit = n = 0
    for x, y, l in loader:
        p = m_sum(model(x.float().to(dev), padded_sequence_mask(l).transpose(0, 1).to(dev))).argmax(1).cpu()
        hit += int((p == y).sum()); n += len(y); functional.reset_net(model)
    return hit / n


best_val, best_ep, best_sd, test_curve = -1, -1, None, []
t0 = time.time()
for ep in range(c.epochs):
    model.train()
    ramp = min(1.0, (ep + 1) / max(1, c.epochs // 2))
    for x, y, l in train_loader:
        out = model(x.float().to(dev), padded_sequence_mask(l).transpose(0, 1).to(dev))
        loss = nn.CrossEntropyLoss()(m_sum(out), F.one_hot(y, c.n_outputs).float().to(dev))
        if LAM > 0:
            loss = loss + LAM * ramp * R.cert_penalty(model.rec)
        opt.zero_grad(); loss.backward(); opt.step(); functional.reset_net(model)
    sched.step()
    va = acc(valid_loader)
    if va > best_val and ep >= SELECT_FROM:
        best_val, best_ep, best_sd = va, ep, copy.deepcopy(model.state_dict())
    te = acc(test_loader) if TEST else float("nan")
    test_curve.append(te)
    print(f"epoch {ep} val {va*100:.2f} best_val {best_val*100:.2f}@{best_ep} ({(time.time()-t0)/(ep+1):.1f}s/ep)", flush=True)
torch.save(best_sd, os.path.expanduser(f"~/research/models/{TAG}_s{c.seed}_Best_ACC.pt"))
model.load_state_dict(best_sd)
cert = R.certify(model, valid_loader, dev, padded_sequence_mask, functional.reset_net)
_ad = R.CERT_ADAPT; R.CERT_ADAPT = not _ad                  # the other certificate, for the record
cert_other = R.certify(model, valid_loader, dev, padded_sequence_mask, functional.reset_net); R.CERT_ADAPT = _ad
res = dict(tag=TAG, seed=c.seed, epochs=c.epochs, lam=LAM, select_from=SELECT_FROM, n_params=int(n_params),
           best_epoch=best_ep, acc_val=best_val, minutes=(time.time() - t0) / 60, cert_other=cert_other, **cert)
if TEST:
    res["acc_test"] = acc(test_loader)
    res["test_best_epoch_rule"] = float(np.nanmax(test_curve))
    res["test_cert"] = R.certify(model, test_loader, dev, padded_sequence_mask, functional.reset_net)
    R.CERT_ADAPT = not _ad
    res["test_cert_other"] = R.certify(model, test_loader, dev, padded_sequence_mask, functional.reset_net)
    R.CERT_ADAPT = _ad
print("RESULT " + json.dumps(res), flush=True)
json.dump(res, open(os.path.expanduser(f"~/research/logs/{TAG}_s{c.seed}.json"), "w"), indent=1)
