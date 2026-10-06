"""PILOT-001 (pre-registered in research/CANDIDATE_STREAMING_ENCODER.md):
when Delta seconds of audio are appended to a T-second buffer, how much do the
Whisper encoder activations of the OLD frames change, per layer and vs distance
from the boundary?

Conditions: model in {base, small}; T in {5, 10, 20} s; Delta = 1 s;
padding in {"pad30" (Whisper's standard 30 s window), "nopad"}.
Encoder frames: 50 per second (after the stride-2 conv).

The HF encoder only accepts exactly 30 s, so a manual forward (conv1, conv2,
positions, blocks, final LayerNorm) is used and verified against HF on 30 s input.
"""
import json, sys
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from datasets import load_dataset

torch.set_grad_enabled(False)
SR, FPS = 16000, 50
MODELS = sys.argv[1:] or ["openai/whisper-base", "openai/whisper-small"]


def audio_stream(min_seconds=45):
    import io, soundfile as sf
    from datasets import Audio
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))      # avoid torchcodec; decode FLAC ourselves
    xs = []
    for a in ds["audio"]:
        y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        assert sr == SR, sr
        xs.append(y)
    x = np.concatenate(xs)
    assert x.size >= min_seconds * SR, x.size / SR
    return x


def encoder_states(enc, feats):
    """Manual Whisper encoder forward for any length; returns [emb, layer1..L, final_ln]."""
    h = torch.nn.functional.gelu(enc.conv1(feats))
    h = torch.nn.functional.gelu(enc.conv2(h)).permute(0, 2, 1)
    h = h + enc.embed_positions.weight[: h.shape[1]]
    out = [h]
    for layer in enc.layers:
        o = layer(h, attention_mask=None)
        h = o[0] if isinstance(o, tuple) else o
        out.append(h)
    out.append(enc.layer_norm(h))
    return [o[0] for o in out]


def feats_for(fe, x, pad30):
    if pad30:
        return fe(x, sampling_rate=SR, return_tensors="pt").input_features
    f = fe(x, sampling_rate=SR, return_tensors="pt", padding="longest",
           truncation=False).input_features
    n = 2 * int(round(x.size / SR * FPS))          # mel frames = 100/s; keep exact length
    return f[..., :n]


def main():
    x_all = audio_stream()
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    results = []
    for name in MODELS:
        model = WhisperModel.from_pretrained(name).eval()
        enc = model.encoder
        # --- verification of the manual forward against HF on a 30 s input
        f30 = feats_for(fe, x_all[: 10 * SR], True)
        ref = enc(f30).last_hidden_state[0]
        mine = encoder_states(enc, f30)[-1]
        err = float((ref - mine).abs().max())
        print(f"{name}: manual vs HF max abs diff = {err:.2e}", flush=True)
        assert err < 1e-3, "manual encoder forward does not match HF"
        for pad in (True, False):
            for T in (5, 10, 20):
                for start in (0, 7, 15):                        # 3 audio segments (seconds)
                    a = x_all[start * SR:(start + T) * SR]
                    b = x_all[start * SR:(start + T + 1) * SR]
                    ha = encoder_states(enc, feats_for(fe, a, pad))
                    hb = encoder_states(enc, feats_for(fe, b, pad))
                    n_old = T * FPS
                    for li, (u, v) in enumerate(zip(ha, hb)):
                        u, v = u[:n_old], v[:n_old]
                        r = ((v - u).norm(dim=-1) / u.norm(dim=-1)).numpy()
                        dist = (n_old - np.arange(n_old)) / FPS     # seconds from boundary
                        bins = {f"d{lo}-{hi}s": float(np.median(r[(dist > lo) & (dist <= hi)]))
                                for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20))
                                if np.any((dist > lo) & (dist <= hi))}
                        results.append(dict(model=name.split("-")[-1], pad="pad30" if pad else "nopad",
                                            T=T, start=start, layer=li, n_layers=len(ha) - 1,
                                            median_rel=float(np.median(r)),
                                            frac_lt_1pct=float(np.mean(r < 0.01)),
                                            frac_lt_5pct=float(np.mean(r < 0.05)), **bins))
                print(f"{name} pad={pad} done", flush=True)
    json.dump(results, open("../results/pilot001.json", "w"), indent=1)

    import pandas as pd
    df = pd.DataFrame(results)
    last = df[df.layer == df.n_layers]                      # final LayerNorm output
    cols = ["median_rel", "frac_lt_1pct", "frac_lt_5pct", "d0-1s", "d1-2s", "d2-5s", "d5-10s"]
    print("\nFINAL encoder output, mean over 3 segments:")
    print(last.groupby(["model", "pad", "T"])[cols].mean().round(4).to_string())
    print("\nPer-layer median relative change (T=10, nopad):")
    print(df[(df["T"] == 10) & (df.pad == "nopad")].groupby(["model", "layer"]).median_rel.mean()
          .round(4).unstack(0).to_string())


if __name__ == "__main__":
    main()
