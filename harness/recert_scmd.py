"""ACC-012c stage 2: re-certify an existing ACC-012b checkpoint on the validation set with both certificates.
No training. Run from ~/research/refs/SCommander-protocol. Env: CKPT, TAG."""
import os, json, torch
from spikingjelly.activation_based import functional
from best_config_SHD_former import Config
from scmd_protocol_data import loaders, padded_sequence_mask
import scmd_rec as R
c = Config(); c.seed = 0; c.backend = "torch"; c.n_hidden_neurons = c.n_hidden_neurons_list[0]; c.hidden_dims = c.n_hidden_neurons
_, va, _ = loaders(c); dev = "cuda"
m = R.SpikCommanderRec(c).to(dev); m.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
out = {}
for flag in (False, True):
    R.CERT_ADAPT = flag
    out["adaptive" if flag else "plain"] = R.certify(m, va, dev, padded_sequence_mask, functional.reset_net)
    x = out["adaptive" if flag else "plain"]
    print(f"{'adaptive' if flag else 'plain   '}: certified {x['core_cert']*100:.2f}%  oracle {x['core_oracle']*100:.2f}%  "
          f"{x['pct_oracle']*100:.1f}% of oracle  viol {x['violations']}", flush=True)
json.dump(out, open(os.path.expanduser(f"~/research/logs/{os.environ['TAG']}.json"), "w"), indent=1)
