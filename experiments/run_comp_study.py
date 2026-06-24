"""Does encoder sharing help when the ENCODER is the bottleneck?

FashionMNIST complementary mode: each modality = a disjoint noisy third of the
image, so a single modality is weak and the three must be fused. Encoders here
matter strongly (trained probe beats random by ~20pp on every modality), and a
vehicle starved on one modality is crippled (fused ~chance) -- exactly the
regime the paper's premise needs. Real SF GPS mobility, gated aggregation
(needed to avoid negative transfer), Local control, K=100, 5 seeds.
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
ROUNDS = int(os.environ.get("COMP_ROUNDS", "100"))
WORKERS = int(os.environ.get("COMP_WORKERS", "6"))   # tiny CNNs -> 6 fit GPU
TRACE = "realSF_n60"
TAG = "encc"
RUN_ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

jobs = [dict(method=m, seed=s) for m in METHODS for s in SEEDS]


def run_job(job):
    m, s = job["method"], job["seed"]
    name = f"{TAG}_{m}_{TRACE}_s{s}"
    log = os.path.join(LOGDIR, name + ".log")
    cmd = [PY, RUNNER, "--method", m, "--trace", TRACE, "--seed", str(s),
           "--rounds", str(ROUNDS), "--tag", TAG, "--dataset", "fmnist",
           "--modality-mode", "complementary", "--comp-noise", "0.7",
           "--fusion-mode", "concat", "--starve-frac", "0.30",
           "--phi-agg", "0.15", "--gated-agg", "--no-gat"]
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=RUN_ENV)
    return f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} ({time.time()-t0:.0f}s)"


if __name__ == "__main__":
    print(f"{len(jobs)} jobs, {WORKERS} workers, {ROUNDS} rounds, "
          f"fmnist-complementary, gated", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for line in ex.map(run_job, jobs):
            print(line, flush=True)
    print("done: complementary study complete", flush=True)
