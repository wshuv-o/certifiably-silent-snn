"""PILOT-002 (pre-registered in research/BRAINSTORM_002.md): after a small code edit in
a long context, how much do the hidden states of the tokens AFTER the edit change
in a small causal LLM? (Premise of edit-aware KV-cache reuse.)
"""
import json, re, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
MODEL = "Qwen/Qwen2.5-0.5B"
CTX = 2048
VENV = "/home/shuvo/research/venv_speech/lib/python3.12/site-packages/"
FILES = [VENV + "numpy/lib/_function_base_impl.py", VENV + "pandas/core/frame.py",
         VENV + "transformers/generation/utils.py"]


def edits(lines, i):
    """Four realistic edits applied at line i; return list of (name, new_lines)."""
    out = []
    ln = lines[i]
    m = re.search(r"\b([a-z_][a-z0-9_]{2,})\b", ln)
    if m:
        out.append(("rename", lines[:i] + [ln[:m.start()] + m.group(1) + "_v2" + ln[m.end():]] + lines[i + 1:]))
    m = re.search(r"\b\d+\b", ln)
    out.append(("literal", lines[:i] + [(ln[:m.start()] + str(int(m.group()) + 7) + ln[m.end():]) if m
                                        else ln + "  # 7"] + lines[i + 1:]))
    indent = re.match(r"\s*", ln).group()
    out.append(("insert", lines[:i] + [indent + "tmp_value = None"] + lines[i:]))
    out.append(("delete", lines[:i] + lines[i + 1:]))
    return out


def states(model, ids):
    o = model(input_ids=ids[None], output_hidden_states=True)
    return [h[0] for h in o.hidden_states], o.logits[0].argmax(-1)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    rows = []
    for f in FILES:
        text = open(f).read()
        lines = text.split("\n")
        # trim to ~CTX tokens by lines
        ids_all = tok(text).input_ids
        keep, acc = [], 0
        for ln in lines:
            n = len(tok(ln + "\n").input_ids)
            if acc + n > CTX: break
            keep.append(ln); acc += n
        # edit line: first non-blank, non-comment line past the middle
        mid = len(keep) // 2
        i = next(k for k in range(mid, len(keep)) if keep[k].strip() and not keep[k].strip().startswith("#"))
        base_ids = torch.tensor(tok("\n".join(keep)).input_ids)
        hb, pb = states(model, base_ids)
        suffix_text = "\n" + "\n".join(keep[i + 1:])
        for name, new in edits(keep, i):
            new_ids = torch.tensor(tok("\n".join(new)).input_ids)
            # align suffix: the tokens after the edited line must be identical in both
            n_suf = len(tok(suffix_text).input_ids) - 2          # conservative: drop boundary tokens
            if n_suf < 64 or not torch.equal(base_ids[-n_suf:], new_ids[-n_suf:]):
                print(f"{f.split('/')[-1]} {name}: suffix tokenization mismatch, skipped"); continue
            he, pe = states(model, new_ids)
            dist = np.arange(n_suf, 0, -1)[::-1]                 # 1..n_suf tokens after the edit region
            agree = (pb[-n_suf:] == pe[-n_suf:]).numpy()
            for li, (u, v) in enumerate(zip(hb, he)):
                u, v = u[-n_suf:], v[-n_suf:]
                r = ((v - u).norm(dim=-1) / u.norm(dim=-1)).numpy()
                far = dist > 32
                rows.append(dict(file=f.split("/")[-1], edit=name, layer=li, n_layers=len(hb) - 1,
                                 n_suffix=int(n_suf), median_far=float(np.median(r[far])),
                                 frac_lt_1pct_far=float(np.mean(r[far] < 0.01)),
                                 frac_lt_2pct_far=float(np.mean(r[far] < 0.02)),
                                 near_0_32=float(np.median(r[~far])),
                                 d32_256=float(np.median(r[(dist > 32) & (dist <= 256)])),
                                 d256_plus=float(np.median(r[dist > 256])) if np.any(dist > 256) else np.nan,
                                 top1_agree_far=float(agree[far].mean())))
            print(f"{f.split('/')[-1]} {name} done (suffix {n_suf} tokens)", flush=True)
    json.dump(rows, open("../results/pilot002.json", "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows)
    last = df[df.layer == df.n_layers]
    cols = ["median_far", "frac_lt_1pct_far", "frac_lt_2pct_far", "near_0_32", "d32_256", "d256_plus", "top1_agree_far"]
    print("\nLAST layer, per file x edit:\n", last[["file", "edit"] + cols].round(4).to_string(index=False))
    print("\nMEAN over files x edits:\n", last[cols].mean().round(4).to_string())
    print("\nPer-layer median_far (mean over cases):\n",
          df.groupby("layer").median_far.mean().round(4).to_string())


if __name__ == "__main__":
    main()
