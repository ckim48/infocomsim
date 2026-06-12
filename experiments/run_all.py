"""Experiment campaign driver: runs the full grid with bounded parallelism."""
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNNER = os.path.join(BASE, "sim", "run_experiment.py")
LOG_DIR = os.path.join(BASE, "results", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
ROUNDS = 55

jobs = []

# 1) main comparison: 3 seeds (trace seed == data seed), medium density
for m, s in itertools.product(METHODS, [1, 2, 3]):
    jobs.append(dict(tag="main", method=m, trace=f"med_n60_s{s}", seed=s))

# 2) V2V range sweep (seed 1; R=200 reused from main)
for m, R in itertools.product(METHODS, [100, 150, 300]):
    jobs.append(dict(tag="sweepR", method=m, trace="med_n60_s1", seed=1,
                     r_v2v=R))

# 3) cache capacity sweep (cache-using methods; C=4 reused from main)
for m, C in itertools.product(
        ["RECD", "LRU-Random", "Mobility-Greedy"], [2, 6, 8]):
    jobs.append(dict(tag="sweepC", method=m, trace="med_n60_s1", seed=1,
                     cache_encoders=C))

# 4) Lyapunov V sweep (RECD only; V=50 reused from main)
for V in [5, 20, 100, 200]:
    jobs.append(dict(tag="sweepV", method="RECD", trace="med_n60_s1", seed=1,
                     V=V))

# 5) number of vehicles sweep
for m, n in itertools.product(METHODS, [30, 100]):
    jobs.append(dict(tag="sweepN", method=m, trace=f"med_n{n}_s1", seed=1))

# 6) background traffic density sweep
for m, d in itertools.product(METHODS, ["low", "high"]):
    jobs.append(dict(tag="sweepD", method=m, trace=f"{d}_n60_s1", seed=1))

# 7) ablation: RECD without GAT (Markov-kernel fallback)
jobs.append(dict(tag="ablation", method="RECD", trace="med_n60_s1", seed=1,
                 no_gat=True))


def run_job(job):
    cmd = [PY, RUNNER,
           "--method", job["method"],
           "--trace", job["trace"],
           "--seed", str(job["seed"]),
           "--rounds", str(ROUNDS),
           "--tag", job["tag"]]
    if "r_v2v" in job:
        cmd += ["--r-v2v", str(job["r_v2v"])]
    if "cache_encoders" in job:
        cmd += ["--cache-encoders", str(job["cache_encoders"])]
    if "V" in job:
        cmd += ["--V", str(job["V"])]
    if job.get("no_gat"):
        cmd += ["--no-gat"]
    if "dataset" in job:
        cmd += ["--dataset", job["dataset"]]
    name = "_".join(
        str(job.get(k, "")) for k in
        ["tag", "method", "trace", "r_v2v", "cache_encoders", "V", "seed"]
    ).replace("__", "_")
    log = os.path.join(LOG_DIR, name + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rc


if __name__ == "__main__":
    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        rcs = list(ex.map(run_job, jobs))
    print(f"done: {sum(1 for r in rcs if r == 0)}/{len(rcs)} succeeded")
