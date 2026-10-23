"""SageMaker training-job entry point for CORE-SCALING-002: runs the many-core engine benchmark on the job's
instance and stores the log in /opt/ml/model (uploaded to S3 as model.tar.gz). No training happens."""
import os, subprocess, sys, shutil
here = os.path.dirname(os.path.abspath(__file__))
out = os.environ.get("SM_MODEL_DIR", "/opt/ml/model"); os.makedirs(out, exist_ok=True)
info = subprocess.run("nproc; lscpu | head -20; free -g; uptime", shell=True, capture_output=True, text=True).stdout
print(info, flush=True); open(f"{out}/host.txt", "w").write(info)
# a freshly booted instance is not idle (load ~2.6 at 1 min uptime): wait for it to settle rather than
# trip run_manycore.sh's idle guard. Poll every 15 s for up to 15 min until the 1-minute load is below 1.0.
import time
for i in range(60):
    l1 = float(open("/proc/loadavg").read().split()[0])
    print(f"waiting for idle: 1-min load {l1:.2f}", flush=True)
    if l1 < 1.0:
        break
    time.sleep(15)
env = dict(os.environ, SRC=f"{here}/data", WORK="/tmp/manycore", REPEATS=os.environ.get("REPEATS", "3"),
           CORE_LIST=os.environ.get("CORE_LIST", "4 8 16 32 64 128"))
with open(f"{out}/" + os.environ.get("LOGNAME_OUT", "manycore.log"), "w") as log:
    p = subprocess.Popen(["bash", f"{here}/" + os.environ.get("SCRIPT", "run_manycore.sh")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
    rc = p.wait()
print("EXIT", rc, flush=True)
sys.exit(rc)
