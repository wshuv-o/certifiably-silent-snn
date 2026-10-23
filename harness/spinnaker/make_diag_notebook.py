"""Write diagnose_spinnaker.ipynb: input pathway only, 5 samples, at real time and slowed down 10x.

The membrane threshold is set out of reach so each recorded v is the leaky sum of the input, which the
notebook compares against the same sum computed in numpy. Uses spinnaker_pack.npz."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md("""# Diagnose the input pathway on SpiNNaker

Runs only the 700 input channels into the 512 neurons for 5 samples, with firing switched off, twice:
at real time and slowed down 10x. Kernel **EBRAINS-25.10**, `spinnaker_pack.npz` in the same folder. Run all
cells, then send back `diag_result.npz` and the printed lines that start with `RESULT`.""")

code("""import numpy as np, time
import pyNN.spiNNaker as sim
P = np.load("spinnaker_pack.npz")
win, X = P["win"].astype(np.float64), P["X"]; T = int(P["T"]); H = win.shape[0]; beta = float(P["beta"])
NS, GAP, T0, TAU_M, BLOCK = 5, 30, 10, 2.0, 128
starts = np.array([T0 + b * (T + GAP) for b in range(NS)]); total = T0 + NS * (T + GAP) + 10
I = np.zeros((total, H))
for b in range(NS): I[starts[b]:starts[b] + T] = X[b] @ win.T
ref = np.zeros_like(I); v = np.zeros(H)
for t in range(total): v = beta * v + I[t]; ref[t] = v""")

code("""def run(tsf):
    sim.setup(timestep=1.0, time_scale_factor=tsf)
    # calibration neuron and the network in the same run, so both see the same settings
    sim.set_number_of_neurons_per_core(sim.IF_curr_delta, 32)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 32)
    model = dict(tau_m=TAU_M, cm=1.0, v_rest=0.0, v_reset=0.0, v_thresh=1e4, tau_refrac=0.0, i_offset=0.0)
    cal = sim.Population(1, sim.IF_curr_delta(**model), label="cal"); cal.initialize(v=0.0)
    cs = sim.Population(1, sim.SpikeSourceArray(spike_times=[[2.0]]), label="cal_src")
    sim.Projection(cs, cal, sim.OneToOneConnector(), sim.StaticSynapse(weight=1.0, delay=1.0), receptor_type="excitatory")
    times = [[] for _ in range(X.shape[2])]
    for b in range(NS):
        tb, cb = np.nonzero(X[b])
        for t, c in zip(tb, cb): times[c].append(float(starts[b] + t))
    inp = sim.Population(X.shape[2], sim.SpikeSourceArray(spike_times=[sorted(t) for t in times]), label="input")
    rec = sim.Population(H, sim.IF_curr_delta(**model), label="recurrent"); rec.initialize(v=0.0)
    for sign, rt in ((1, "excitatory"), (-1, "inhibitory")):
        for b0 in range(0, H, BLOCK):
            post_i, pre_j = np.nonzero(sign * win[b0:b0 + BLOCK] > 0)
            conn = [(int(j), int(b0 + i), float(abs(win[b0 + i, j])), 1.0) for i, j in zip(post_i, pre_j)]
            sim.Projection(inp, rec, sim.FromListConnector(conn), receptor_type=rt)
    rec.record("v"); cal.record("v")
    t0 = time.time(); sim.run(total); took = time.time() - t0
    v = np.array(rec.get_data("v").segments[0].filter(name="v")[0]).astype(np.float64)
    vc = np.array(cal.get_data("v").segments[0].filter(name="v")[0]).ravel()
    sim.end()
    return v, vc, took

def score(v, vc, tsf):
    k = int(np.argmax(np.abs(vc[2:]) > 1e-6)) + 2; gain = vc[k]
    best = None
    for lag in range(0, 5):
        a = ref[:total - lag]; b = v[lag:] / gain
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]; slope = (a * b).sum() / (a * a).sum()
        if best is None or c > best[1]: best = (lag, c, slope)
    lag, c, slope = best
    ex = ref[:total - lag] > 0; inh = ref[:total - lag] < 0; b = v[lag:] / gain
    print(f"RESULT tsf={tsf} gain={gain:.4f} arrive={k - 2} lag={lag} corr={c:.4f} slope={slope:.3f} "
          f"slope_pos={(ref[:total-lag][ex] * b[ex]).sum() / (ref[:total-lag][ex] ** 2).sum():.3f} "
          f"slope_neg={(ref[:total-lag][inh] * b[inh]).sum() / (ref[:total-lag][inh] ** 2).sum():.3f} "
          f"vmin={v.min():.2f} vmax={v.max():.2f} refmin={ref.min():.2f} refmax={ref.max():.2f}")
    return gain, k - 2, lag, c""")

code("""out = {}
for tsf in (1, 10):
    v, vc, took = run(tsf)
    print("tsf", tsf, "run took", round(took, 1), "s")
    out[f"v{tsf}"] = v.astype(np.float32); out[f"cal{tsf}"] = vc
    out[f"score{tsf}"] = np.array(score(v, vc, tsf))
np.savez_compressed("diag_result.npz", **out)
print("saved diag_result.npz")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "EBRAINS-25.10", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(os.path.join(HERE, "diagnose_spinnaker.ipynb"), "w"), indent=1)
print("written diagnose_spinnaker.ipynb")
