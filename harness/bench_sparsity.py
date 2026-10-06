"""EXP-001: measure the dense-sync vs sparse-sync crossover (the sparsity
component) on CPU and GPU, and record raw timings for fitting cost constants.

Protocol (v2, after the pilot showed laptop GPU clock swings of 300-1860 MHz):
  * DS and SS are interleaved rep by rep (paired design), with the order
    alternating each rep, so both modes see the same thermal/clock state.
  * Before each cell the GPU is warmed up with untimed steps.
  * The GPU SM clock is logged on every rep (outside the timed region).

Usage (inside WSL, venv active):
  python bench_sparsity.py --N 1048576 --F 8 32 64 --windows 256 0 --out ../results/exp001
(window 0 means uniformly random destinations)
"""
import argparse, json, os, platform, time
# CPU protocol (diagnosed 2026-10-04): under WSL, 12 threads (with SMT) and
# passive OpenMP waiting give ~20 ms wake-up stalls on sub-ms steps. One thread
# per physical core plus active waiting keeps the median within ~1.3x of the min.
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
os.environ.setdefault("OMP_WAIT_POLICY", "active")
import numpy as np
from numba import get_num_threads
import numba

from graphgen import make_graph, sample_active, reference_step, owner_imbalance
from cpu_kernels import CPUEngine

S_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
VERIFY_S = {1e-3, 0.1, 1.0}
THETA = 1.0
MODES = ("DS", "SS")

_nvml = None


def sm_clock():
    """Current GPU SM clock in MHz (best effort; -1 if unavailable)."""
    global _nvml
    try:
        import pynvml as nv
        if _nvml is None:
            nv.nvmlInit()
            _nvml = nv.nvmlDeviceGetHandleByIndex(0)
        return nv.nvmlDeviceGetClockInfo(_nvml, nv.NVML_CLOCK_SM)
    except Exception:
        return -1


def gpu_telemetry():
    try:
        import pynvml as nv
        sm_clock()
        return {"sm_clock_mhz": nv.nvmlDeviceGetClockInfo(_nvml, nv.NVML_CLOCK_SM),
                "temp_c": nv.nvmlDeviceGetTemperature(_nvml, nv.NVML_TEMPERATURE_GPU),
                "power_w": nv.nvmlDeviceGetPowerUsage(_nvml) / 1000.0}
    except Exception as e:  # telemetry is best-effort, never fatal
        return {"error": str(e)}


def check(state, nxt, ref_state, ref_nxt, N):
    ok_state = np.allclose(state, ref_state, rtol=1e-4, atol=1e-4)
    a, b = set(np.asarray(nxt).tolist()), set(ref_nxt.tolist())
    mismatch = len(a ^ b)  # threshold ties can flip under different summation order
    return ok_state and mismatch <= max(1, int(1e-4 * N)), mismatch


class CPUStepper:
    def __init__(self, eng, active, state0):
        self.eng, self.active, self.state0 = eng, active, state0
        self.state = state0.copy()

    def step(self, mode):
        """Restore state, run one timed step; returns (seconds, state, next-frontier)."""
        fn = self.eng.ds if mode == "DS" else self.eng.ss
        np.copyto(self.state, self.state0)
        t0 = time.perf_counter()
        nxt = fn(self.active, self.state, THETA)
        return time.perf_counter() - t0, self.state, nxt

    def clock(self):
        return -1


class GPUStepper:
    def __init__(self, eng, active_h, state0_h):
        import cupy as cp
        self.cp, self.eng = cp, eng
        self.active = cp.asarray(active_h)
        self.state0 = cp.asarray(state0_h)
        self.state = self.state0.copy()
        self.ev0, self.ev1 = cp.cuda.Event(), cp.cuda.Event()

    def step(self, mode):
        cp = self.cp
        fn = self.eng.ds if mode == "DS" else self.eng.ss
        cp.copyto(self.state, self.state0)
        self.ev0.record()
        out = fn(self.active, self.state, THETA)
        self.ev1.record()
        self.ev1.synchronize()
        return cp.cuda.get_elapsed_time(self.ev0, self.ev1) / 1e3, self.state, out

    def warm_up(self, seconds=0.3):
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            self.step("DS")

    def clock(self):
        return sm_clock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1 << 20)
    ap.add_argument("--F", type=int, nargs="+", default=[8, 32, 64])
    ap.add_argument("--windows", type=int, nargs="+", default=[256, 0])
    ap.add_argument("--s", type=float, nargs="+", default=S_GRID)
    ap.add_argument("--platforms", nargs="+", default=["cpu", "gpu"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spread", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out", default="../results/exp001")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    meta = {"exp": "EXP-001", "protocol": "v2-interleaved", "args": vars(a), "theta": THETA,
            "started": time.ctime(), "python": platform.python_version(), "numpy": np.__version__,
            "numba": numba.__version__, "numba_threads": get_num_threads(),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "uname": platform.uname()._asdict(), "telemetry": []}
    if "gpu" in a.platforms:
        import cupy as cp
        from gpu_kernels import GPUEngine
        meta["cupy"] = cp.__version__
        meta["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    rows, fails = [], []
    rng_state = np.random.default_rng(a.seed + 1)
    for F in a.F:
        for win in a.windows:
            window = a.N if win == 0 else win
            dst, w, in_ptr, in_idx, in_w = make_graph(a.N, F, window, seed=a.seed)
            state0 = rng_state.uniform(0.0, 1.0, a.N).astype(np.float32)
            engines = {}
            if "cpu" in a.platforms:
                engines["cpu"] = CPUEngine(a.N, F, dst, w, in_ptr, in_idx, in_w)
            if "gpu" in a.platforms:
                engines["gpu"] = GPUEngine(a.N, F, dst, w, in_ptr, in_idx, in_w)
            rng_act = np.random.default_rng(a.seed + 2)
            for spread, s in [(sp, s) for sp in a.spread for s in a.s]:
                active = sample_active(a.N, s, rng_act, spread)
                kappa = owner_imbalance(active, dst, F, a.N, get_num_threads())
                for plat, eng in engines.items():
                    st = CPUStepper(eng, active, state0) if plat == "cpu" else GPUStepper(eng, active, state0)
                    if plat == "gpu":
                        st.warm_up()
                        meta["telemetry"].append({"F": F, "win": win, "s": s, **gpu_telemetry()})
                    times = {m: [] for m in MODES}
                    for r in range(a.warm + a.reps):
                        order = MODES if r % 2 == 0 else MODES[::-1]
                        for mode in order:
                            clk = st.clock()
                            t, _, _ = st.step(mode)
                            if r >= a.warm:
                                times[mode].append(t)
                                rows.append((plat, mode, a.N, F, win, s, active.size,
                                             r - a.warm, t, clk, spread, kappa))
                    if s in VERIFY_S:  # untimed verification against the NumPy reference
                        ref = reference_step(active, dst, w, F, state0, THETA)
                        for mode in MODES:
                            _, state, out = st.step(mode)
                            if plat == "gpu":
                                state, out = cp.asnumpy(state), eng.result(out)
                            ok, mm = check(state, out, ref[0], ref[1], a.N)
                            if not ok:
                                fails.append((plat, mode, F, win, s, mm))
                    print(f"{plat} F={F:3d} win={win:4d} spread={spread:<5g} kappa={kappa:5.2f} s={s:<7g} "
                          f"DS={np.median(times['DS'])*1e3:9.3f} ms  SS={np.median(times['SS'])*1e3:9.3f} ms",
                          flush=True)
            del engines
            if "gpu" in a.platforms:
                cp.get_default_memory_pool().free_all_blocks()

    with open(a.out + ".csv", "w") as f:
        f.write("platform,mode,N,F,window,s,m,rep,time_s,sm_clock_mhz,spread,kappa\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    meta["finished"] = time.ctime()
    meta["verification_failures"] = fails
    with open(a.out + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("VERIFICATION:", "PASS" if not fails else f"FAIL {fails}")


if __name__ == "__main__":
    main()

