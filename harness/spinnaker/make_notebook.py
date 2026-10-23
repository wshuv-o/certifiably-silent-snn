"""Write run_on_spinnaker.ipynb for the EBRAINS lab (kernel EBRAINS-25.10)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md("""# Certifiably silent SNN on SpiNNaker

Runs the trained locally connected network (512 neurons, 16 cores of 32, delays 2/4/8 ms) on 50 SHD test samples.
SpiNNaker's `IF_curr_delta` neuron is used: leaky integrate-and-fire with instantaneous synapses, hard reset, no
threshold adaptation. Upload `spinnaker_pack.npz` to the same folder as this notebook, select the kernel
**EBRAINS-25.10**, and run all cells. At the end, download `spinnaker_result.npz` and send it back.""")

code("""import numpy as np, time
import pyNN.spiNNaker as sim
P = np.load("spinnaker_pack.npz")
win, W, X, y = P["win"], P["W"], P["X"], P["y"]
delays = [int(d) for d in P["delays"]]; T = int(P["T"]); H = W.shape[1]; NS = X.shape[0]
TAU_M, GAP, T0 = 2.0, 30, 10
TSF = 100   # run 100x slower than real time so every core can take up to ~450 input spikes per step
print("samples", NS, "neurons", H, "delays", delays)""")

md("## 1. Calibration: how one input spike moves the membrane")
code("""def calibrate(cm, w=0.25):
    sim.setup(timestep=1.0, time_scale_factor=TSF)
    n = sim.Population(1, sim.IF_curr_delta(tau_m=TAU_M, cm=cm, v_rest=0.0, v_reset=0.0, v_thresh=1000.0,
                                            tau_refrac=0.0, i_offset=0.0))
    n.initialize(v=0.0)                                   # SpiNNaker starts membranes at -65 mV by default
    s = sim.Population(1, sim.SpikeSourceArray(spike_times=[[5.0]]))
    sim.Projection(s, n, sim.OneToOneConnector(), sim.StaticSynapse(weight=w, delay=1.0), receptor_type="excitatory")
    n.record("v"); sim.run(20)
    v = np.array(n.get_data("v").segments[0].filter(name="v")[0]).ravel(); sim.end()
    after = np.abs(v[5:]) > 1e-6
    assert after.any(), 'input spike had no effect on the membrane'
    k = 5 + int(np.argmax(after))                         # first step the test spike moves the membrane
    return v, k, v[k] / w, v[k + 1] / v[k]

v, k, gain, ratio = calibrate(1.0)
print("v trace", np.round(v[:14], 5))
print("first nonzero index", k, "(spike sent at 5 ms, delay 1 ms)")
print("gain per unit weight", round(float(gain), 5), " decay ratio", round(float(ratio), 5), " target", round(float(np.exp(-1 / TAU_M)), 5))
assert gain > 0 and 0 < k - 5 <= 3, f'calibration failed: gain {gain}, arrival {k - 5}; do not run the network'
SCALE = 1.0 / gain          # weights are multiplied by SCALE so one unit of weight moves v by one unit
ARRIVE = k - 5               # steps from spike time to first effect on v, for a 1 ms delay""")

md("## 1b. Calibration: when a neuron's own spike reaches another neuron")
code("""def calibrate_neuron():
    sim.setup(timestep=1.0, time_scale_factor=TSF)
    model = dict(tau_m=TAU_M, cm=1.0, v_rest=0.0, v_reset=0.0, tau_refrac=0.0, i_offset=0.0)
    a = sim.Population(1, sim.IF_curr_delta(v_thresh=1.0, **model)); a.initialize(v=0.0)
    b = sim.Population(1, sim.IF_curr_delta(v_thresh=1000.0, **model)); b.initialize(v=0.0)
    s = sim.Population(1, sim.SpikeSourceArray(spike_times=[[5.0]]))
    sim.Projection(s, a, sim.OneToOneConnector(), sim.StaticSynapse(weight=5.0 * SCALE, delay=1.0), receptor_type="excitatory")
    sim.Projection(a, b, sim.OneToOneConnector(), sim.StaticSynapse(weight=0.25 * SCALE, delay=1.0), receptor_type="excitatory")
    a.record(["spikes", "v"]); b.record("v"); sim.run(20)
    sa = a.get_data(["spikes", "v"]).segments[0]; va = np.array(sa.filter(name="v")[0]).ravel()
    tr = int(np.array(sa.spiketrains[0].times)[0]); vb = np.array(b.get_data("v").segments[0].filter(name="v")[0]).ravel()
    sim.end()
    drive = 5 + ARRIVE                                    # step at which the input reaches neuron a
    reset = drive - tr                                    # recorded spike time is this many steps before the crossing
    e1 = int(np.argmax(np.abs(vb) > 1e-6)) - tr           # first effect on b after a 1 ms delay, from the recorded spike
    return va, vb, tr, reset, e1

va, vb, tr, RESET, E1 = calibrate_neuron()
print("a v", np.round(va[:12], 3)); print("b v", np.round(vb[:12], 3))
print("recorded spike", tr, " crossing at recorded+", RESET, " effect on b at recorded+", E1, "for 1 ms delay")
assert np.abs(vb).max() > 0 and 0 <= RESET <= 2 and 1 <= E1 <= 4, "neuron calibration failed; do not run the network"
DSHIFT = RESET - E1 + 1       # hardware delay = d + DSHIFT puts the effect d steps after the crossing
print("RESULT timing: RESET", RESET, "E1", E1, "DSHIFT", DSHIFT, "-> hardware delays", [d + DSHIFT for d in delays])""")

md("## 2. Build and run the network")
code("""total = T0 + NS * (T + GAP) + 10
starts = np.array([T0 + b * (T + GAP) for b in range(NS)])
times = [[] for _ in range(X.shape[2])]
for b in range(NS):
    tb, cb = np.nonzero(X[b])
    for t, c in zip(tb, cb):
        st = starts[b] + t - ARRIVE          # with a 1 ms delay, input of step t reaches the membrane at step t
        if st >= 1: times[c].append(float(st))
times = [sorted(t) for t in times]

sim.setup(timestep=1.0, time_scale_factor=TSF)
try:   # load with the standard loader (small packets); the fast loader's bulk upload exceeds the server's size limit
    from spinn_utilities.config_holder import set_config
    set_config("Machine", "enable_advanced_monitor_support", "False")
    set_config("Java", "use_java", "False")          # Python loader writes memory in small pieces
    print("standard Python loader selected")
    # the server rejects any single upload above about 1 MB, so split every memory write into 256 KB pieces
    from spinnman.spalloc import spalloc_client as _sc
    if not getattr(_sc._SpallocJob.write_data, "_chunked", False):
        _orig_write = _sc._SpallocJob.write_data
        def _chunked_write(self, x, y, address, data, _CH=256 * 1024):
            if len(data) > _CH:
                print(f"  splitting a {len(data) / 1e6:.2f} MB write to chip ({x},{y}) into {-(-len(data) // _CH)} pieces")
            for o in range(0, len(data), _CH):
                _orig_write(self, x, y, address + o, data[o:o + _CH])
        _chunked_write._chunked = True
        _sc._SpallocJob.write_data = _chunked_write
    print("uploads split into 256 KB pieces")
except Exception as e:
    print("could not switch loader:", e)
sim.set_number_of_neurons_per_core(sim.IF_curr_delta, 16)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 32)   # small per-core uploads (server limit ~1 MB per upload)
model = sim.IF_curr_delta(tau_m=TAU_M, cm=1.0, v_rest=0.0, v_reset=0.0, v_thresh=1.0, tau_refrac=0.0, i_offset=0.0)
# busy steps deliver up to ~450 input spikes at once: a larger incoming buffer and more ring-buffer headroom
# keep the summed inhibitory input from being cut off on those steps
try:
    rec = sim.Population(H, model, label="recurrent",
                         additional_parameters={"incoming_spike_buffer_size": 4096, "ring_buffer_sigma": 20.0})
    print("recurrent population: spike buffer 4096, ring buffer sigma 20")
except Exception as e:
    print("additional parameters not accepted (", e, "), using defaults")
    rec = sim.Population(H, model, label="recurrent")
rec.initialize(v=0.0)
inp = sim.Population(X.shape[2], sim.SpikeSourceArray(spike_times=times), label="input")

PROJS = []    # (projection, source label, sign), for reading the stored weights back after the run
BLOCK = 128   # SpiNNaker allows at most 256 synapses per source row, so each projection covers 128 targets
def connect(pre, w, delay, label):
    for sign, rt in ((1, "excitatory"), (-1, "inhibitory")):
        n = 0
        for b0 in range(0, w.shape[0], BLOCK):
            post_i, pre_j = np.nonzero(sign * w[b0:b0 + BLOCK] > 0)
            conn = [(int(j), int(b0 + i), float(abs(w[b0 + i, j]) * SCALE), float(delay)) for i, j in zip(post_i, pre_j)]
            if conn:
                PROJS.append((sim.Projection(pre, rec, sim.FromListConnector(conn), receptor_type=rt, label=f"{label}_{rt}_{b0}"),
                              label, sign))
            n += len(conn)
        print(label, rt, n)

connect(inp, win, 1, "input")
for wi, d in enumerate(delays):
    assert d + DSHIFT >= 1
    connect(rec, W[wi], d + DSHIFT, f"rec_d{d}")   # a crossing at step t reaches v at step t + d
rec.record(["spikes", "v"])
t_start = time.time(); sim.run(total); print("run took", round(time.time() - t_start, 1), "s")""")

md("## 3. Save the recordings")
code("""seg = rec.get_data(["spikes", "v"]).segments[0]
v = np.array(seg.filter(name="v")[0]).astype(np.float32)          # (time, neuron)
spk_i, spk_t = [], []
for st in seg.spiketrains:
    i = int(st.annotations["source_index"]); ts = np.array(st.times).astype(np.int32)
    spk_i += [i] * len(ts); spk_t += list(ts)
np.savez_compressed("spinnaker_result.npz", v=v, spk_i=np.array(spk_i, np.int32), spk_t=np.array(spk_t, np.int32),
                    starts=starts, scale=SCALE, arrive=ARRIVE, gain=gain, ratio=ratio, T=T, gap=GAP,
                    reset=RESET, e1=E1, dshift=DSHIFT, tsf=TSF)
print("spikes", len(spk_t), "v shape", v.shape)

# the weights as stored on the chip (after SpiNNaker's fixed-point rounding), in the network's own units
win_hw = np.zeros_like(win); W_hw = np.zeros_like(W); t0 = time.time()
for proj, label, sign in PROJS:
    for j, i, w in proj.get(["weight"], format="list"):
        if label == "input": win_hw[int(i), int(j)] = sign * w / SCALE
        else: W_hw[delays.index(int(label.split("_d")[1])), int(i), int(j)] = sign * w / SCALE
sim.end()
print("weights read back in", round(time.time() - t0, 1), "s; max relative difference",
      round(float(np.abs(win_hw - win).max() / np.abs(win).max()), 4), round(float(np.abs(W_hw - W).max() / np.abs(W).max()), 4))
np.savez_compressed("spinnaker_result.npz", v=v, spk_i=np.array(spk_i, np.int32), spk_t=np.array(spk_t, np.int32),
                    starts=starts, scale=SCALE, arrive=ARRIVE, gain=gain, ratio=ratio, T=T, gap=GAP,
                    reset=RESET, e1=E1, dshift=DSHIFT, tsf=TSF, win_hw=win_hw.astype(np.float32), W_hw=W_hw.astype(np.float32))
print("-> download spinnaker_result.npz")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "EBRAINS-25.10", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(os.path.join(HERE, "run_on_spinnaker.ipynb"), "w"), indent=1)
print("written run_on_spinnaker.ipynb")
