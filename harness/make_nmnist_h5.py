"""Convert N-MNIST into the SHD/SSC spike-list layout so the existing loader reads it unchanged.

N-MNIST is event-camera vision: a DVS views MNIST digits through three saccades, 34x34 pixels and
two polarities. Writing it as (times, units) variable-length lists, with times normalised to [0,1]
per sample, lets `LazySpikes` bin it exactly as it bins the audio sets, so the temporal protocol is
identical across all three datasets and nothing dataset-specific enters the training path.

    unit = (y * 34 + x) * 2 + p        NIN = 34 * 34 * 2 = 2312

    python harness/make_nmnist_h5.py
"""
import os
import time

import h5py
import numpy as np
import tonic

W = H = 34
NIN = W * H * 2
OUT = os.path.expanduser("~/research/data/nmnist")


def convert(train):
    split = "train" if train else "test"
    dst = os.path.join(OUT, "nmnist_" + split + ".h5")
    if os.path.exists(dst):
        print("exists, skipping: " + dst)
        return
    ds = tonic.datasets.NMNIST(save_to=OUT, train=train)
    n = len(ds)
    print(split + ": " + str(n) + " samples -> " + dst, flush=True)

    vt = h5py.special_dtype(vlen=np.dtype("float32"))
    vu = h5py.special_dtype(vlen=np.dtype("int32"))
    tmp = dst + ".part"
    t0 = time.time()
    with h5py.File(tmp, "w") as f:
        g = f.create_group("spikes")
        dt = g.create_dataset("times", (n,), dtype=vt)
        du = g.create_dataset("units", (n,), dtype=vu)
        lab = np.zeros(n, dtype=np.int64)
        for i in range(n):
            ev, y = ds[i]
            x = ev["x"].astype(np.int32)
            yy = ev["y"].astype(np.int32)
            p = ev["p"].astype(np.int32)
            t = ev["t"].astype(np.float64)
            # clip stray coordinates rather than dropping them silently
            np.clip(x, 0, W - 1, out=x)
            np.clip(yy, 0, H - 1, out=yy)
            p = (p > 0).astype(np.int32)
            span = t.max() - t.min()
            tn = (t - t.min()) / span if span > 0 else np.zeros_like(t)
            dt[i] = tn.astype(np.float32)
            du[i] = ((yy * W + x) * 2 + p).astype(np.int32)
            lab[i] = int(y)
            if i % 5000 == 0:
                print("  %6d/%d  %.0fs" % (i, n, time.time() - t0), flush=True)
        f.create_dataset("labels", data=lab)
        f.attrs["nin"] = NIN
    os.replace(tmp, dst)
    sz = os.path.getsize(dst) / 1e6
    print("wrote %s  %.0f MB  %.0fs" % (dst, sz, time.time() - t0), flush=True)


if __name__ == "__main__":
    convert(False)      # test first: smaller, so a format error surfaces quickly
    convert(True)
