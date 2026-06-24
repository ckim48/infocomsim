"""Quality-heterogeneity regime: the setting where encoder sharing CAN help.

Vehicles keep FULL data (so the local fusion has enough joint samples -- not the
bottleneck), but a fraction have a heavily degraded (noisy) sensor for one
modality, so that modality's locally-trained encoder genuinely fails. A vehicle
that receives a clean encoder for its noisy modality can then improve. Uses
UNGATED aggregation (with full local data, blending a clean donor helps; the
learning-gain gate instead rejects useful donors here). Complementary
FashionMNIST, real SF GPS, K=100, 5 seeds, no quantity starvation.
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
ROUNDS = int(os.environ.get("QUAL_ROUNDS", "100"))
WORKERS = int(os.environ.get("QUAL_WORKERS", "6"))
TRACE = "realSF_n60"
TAG = os.environ.get("QUAL_TAG", "encq")   # encq2 = quality-aware need_weight
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
           "--p-mod", "0.85", "--q-low-frac", "0.5", "--q-low-hi", "0.25",
           "--qual-noise", "2.0", "--starve-frac", "0.0", "--max-out", "1",
           "--fusion-mode", "concat", "--phi-agg", "0.15", "--no-gat"]
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=RUN_ENV)
    return f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} ({time.time()-t0:.0f}s)"


if __name__ == "__main__":
    print(f"{len(jobs)} jobs, {WORKERS} workers, {ROUNDS} rounds, "
          f"harsh-quality (full data, ungated)", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for line in ex.map(run_job, jobs):
            print(line, flush=True)
    print("done: quality study complete", flush=True)
