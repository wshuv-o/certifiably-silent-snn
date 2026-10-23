"""ACC-005 driver, run from inside ~/research/refs/SNN-delays-protocol (see run_acc005.sh).

Control runs (CERT_LAMBDA=0): the reference recipe, best epoch chosen on the speaker-disjoint validation set.
Constrained runs (CERT_LAMBDA>0, INIT=control checkpoint): fine-tune with the certificate penalty; the
final epoch is the model (as in the paper's fine-tunes). The test set is loaded only when TEST=1, and is
evaluated once, on the selected model, after training.
"""
import os, sys, json, time
import torch
from config import Config
from protocol_data import loaders
if float(os.environ.get("ADV", "0")) > 0:
    from protocol_spk import loaders          # ACC-009: training batches carry the speaker index
import acc005_rec as R

E = os.environ.get
TAG = E("TAG", "a5"); SEED = int(E("SEED", "0")); TEST = E("TEST", "0") == "1"; INIT = E("INIT", "")
c = Config
c.seed = SEED
if E("NBINS"):
    # ACC-008: input frequency resolution (reference sums 5 adjacent cochlear channels: 700 -> 140)
    c.n_bins = int(E("NBINS")); c.n_inputs = 700 // c.n_bins
c.epochs = int(E("EPOCHS", str(c.epochs)))
if INIT:
    # fine-tune: delays already resolved (sigma held at its final value, positions frozen), weights at a
    # one-cycle schedule peaking at FT_MAX_LR
    c.final_epoch = 0
    c.lr_w = float(E("FT_LR", "1e-4")); c.max_lr_w = float(E("FT_MAX_LR", "5e-4"))
    c.lr_pos = 0.0; c.max_lr_pos = 0.0
c.t_max_w = c.epochs; c.t_max_pos = c.epochs
c.select_from_epoch = int(E("SELECT_FROM", "0"))   # ACC-007: best-val checkpoint only from this epoch on
if not INIT:
    c.final_epoch = c.epochs // 4
M = os.path.expanduser("~/research/models")
c.save_model_path = f"{M}/{TAG}_s{SEED}_REPL.pt"
dev = "cuda"

train_loader, valid_loader, test_loader = loaders(c)
model = R.SnnDelaysRec(c).to(dev)
if INIT:
    model.load_state_dict(torch.load(INIT, map_location=dev), strict=True)
    if E("FREEZE_POS", "1") == "1":
        for p in model.positions:
            p.requires_grad = False
n_params = sum(p.numel() for p in model.parameters())
print(f"ACC-005 {TAG} seed {SEED} lambda {R.LAM} rec_drop {R.REC_DROP} params {n_params}", flush=True)
t0 = time.time()
# TEST runs also log test accuracy per epoch so the reference's reporting rule (best test epoch) can be
# shown beside ours; selection (Best_ACC checkpoint, final epoch for fine-tunes) never looks at it.
model.train_model(train_loader, valid_loader, test_loader if TEST else None, dev)
torch.save(model.state_dict(), c.save_model_path.replace("REPL", "Final"))
SELECT = E("SELECT", "best_val" if not INIT else "final")
if SELECT == "best_val":
    model.load_state_dict(torch.load(c.save_model_path.replace("REPL", "Best_ACC"), map_location=dev))
sel = SELECT + (f"_from_{c.select_from_epoch}" if getattr(c, "select_from_epoch", 0) else "")
model.reset_model(train=False)
acc_val = R.accuracy(model, valid_loader, dev)
cert = R.certify(model, valid_loader, dev)
pc = R.per_class(model, valid_loader, dev)
res = dict(tag=TAG, seed=SEED, lam=R.LAM, rec_drop=R.REC_DROP, rec_h=R.REC_H, delays=R.DELAYS, K=R.K,
           epochs=c.epochs, selected=sel, n_params=int(n_params), acc_val=acc_val,
           minutes=(time.time() - t0) / 60, val_per_class=pc, adv=R.ADV, **cert)
if TEST:
    res["acc_test"] = R.accuracy(model, test_loader, dev)
    res["test_cert"] = R.certify(model, test_loader, dev)
print("RESULT " + json.dumps(res), flush=True)
json.dump(res, open(os.path.expanduser(f"~/research/logs/{TAG}_s{SEED}.json"), "w"), indent=1)
