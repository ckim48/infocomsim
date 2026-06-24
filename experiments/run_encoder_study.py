"""Encoder-level study on REAL San Francisco taxi GPS mobility (cabspotting).

Question: does V2V encoder sharing improve the *encoder* (per-modality
representation), even though it did not improve the fused model? We use the
Raymobtime 256-beam task (where encoder quality genuinely matters; the 8-sector
task is too easy -- a random encoder probes as well as a trained one), the real
SF GPS trace (real mobility; coverage does NOT trivially saturate as it does
with the SUMO traces), and compare against a Local (no-sharing) control. The
metric per run is encoder_probe_acc (frozen encoder + fresh global linear probe).

25 runs = 5 methods x 5 seeds. ~4 workers, single CPU thread each.
"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/anaconda3/envs/cm-pfl/bin/python")
RUNNER = os.path.join(BASE, "sim", "run_experiment.py")
LOGDIR = os.path.join(BASE, "results", "logs")
os.makedirs(LOGDIR, exist_ok=True)

METHODS = ["Local", "RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
SEEDS = [1, 2, 3, 4, 5]
ROUNDS = int(os.environ.get("ENC_ROUNDS", "100"))
WORKERS = int(os.environ.get("ENC_WORKERS", "4"))
TRACE = "realSF_n60"
GATED = os.environ.get("ENC_GATED", "0") == "1"   # learning-gain-gated aggregation
TAG = "encg" if GATED else "enc"
RUN_ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

jobs = []
for m in METHODS:
    for s in SEEDS:
        jobs.append(dict(method=m, seed=s))


def run_job(job):
    m, s = job["method"], job["seed"]
    name = f"{TAG}_{m}_{TRACE}_s{s}"
    log = os.path.join(LOGDIR, name + ".log")
    cmd = [PY, RUNNER, "--method", m, "--trace", TRACE, "--seed", str(s),
           "--rounds", str(ROUNDS), "--tag", TAG, "--dataset", "raymob",
           "--fusion-mode", "concat", "--starve-frac", "0.30",
           "--phi-agg", "0.15", "--no-gat"] + (["--gated-agg"] if GATED else [])
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=RUN_ENV)
    return f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} ({time.time()-t0:.0f}s)"


if __name__ == "__main__":
    print(f"{len(jobs)} jobs, {WORKERS} workers, {ROUNDS} rounds, trace={TRACE}",
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for line in ex.map(run_job, jobs):
            print(line, flush=True)
    print("done: encoder study complete", flush=True)
