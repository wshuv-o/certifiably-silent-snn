"""ACC-012c step 3: re-certify the ACC-007 certified models (DCLS host) with both certificates, test and
validation. No training. Run from ~/research/refs/SNN-delays-protocol."""
import os, json, torch
from config import Config
from protocol_data import loaders
import acc005_rec as R
c = Config; dev = "cuda"
_, va, te = loaders(c)
out = {}
for s in (1, 2, 3):
    c.seed = s
    m = R.SnnDelaysRec(c).to(dev)
    m.load_state_dict(torch.load(os.path.expanduser(f"~/research/models/a7_C1Tc_s{s}_Best_ACC.pt"), map_location=dev))
    for split, ld in (("val", va), ("test", te)):
        for flag in (False, True):
            R.CERT_ADAPT = flag; x = R.certify(m, ld, dev); k = f"s{s}_{split}_{'adaptive' if flag else 'plain'}"
            out[k] = x
            print(f"{k}: {x['core_cert']*100:.2f}% / {x['core_oracle']*100:.2f}% = {x['pct_oracle']*100:.1f}% of oracle, viol {x['violations']}", flush=True)
json.dump(out, open(os.path.expanduser("~/research/logs/a12c_recert_acc007.json"), "w"), indent=1)
