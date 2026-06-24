"""Paper campaign: main comparison + core sweeps on BOTH testbeds.

Two datasets are run at the paper's K=100 rounds:
  * UCI HAR  (primary, genuinely complementary sensor modalities): concat
    fusion + 20% modality starvation so a starved vehicle can only recover a
    useful encoder via V2V dissemination -- the regime where sharing matters.
  * FashionMNIST (contrast, weakly complementary synthetic image crops):
    legacy mean fusion, no starvation.

For each dataset:
  main      4 methods x 3 seeds      (RECD uses the hierarchical GAT predictor)
  sweepR    V2V range  {100,150,300}
  sweepC    cache cap  {2,6,8}       (cache-using methods only)
  sweepD    bg density {low,high}
Sweeps use the Markov-kernel mobility fallback (no-gat) for speed; the memory
note records it is numerically equivalent to the GAT predictor here.

Run:  ~/anaconda3/envs/cm-pfl/bin/python experiments/run_paper.py
"""
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
CACHE_METHODS = ["RECD", "LRU-Random", "Mobility-Greedy"]
ROUNDS = int(os.environ.get("PAPER_ROUNDS", "100"))
# 4 workers: the Raymobtime LiDAR encoder uses ~1.7GB GPU each, so 6 overflows
# the 10GB card (CUDA OOM). 4 x ~1.7GB fits with headroom.
WORKERS = int(os.environ.get("PAPER_WORKERS", "4"))

# Each worker is pinned to a single CPU thread: with 60 tiny per-vehicle models
# the default OpenMP oversubscription thrashes (user time ~7x wall). One thread
# per process drops a round from ~28s to ~4.5s, and the tiny models leave the
# GPU underutilized so several single-thread processes share it well.
# expandable_segments curbs CUDA fragmentation across the concurrent processes.
RUN_ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

# Testbed: Raymobtime vehicular multimodal beam selection (GPS coord + camera
# image + LiDAR cube -> best Tx beam sector, 8 classes). concat fusion + 20%
# modality starvation so a starved vehicle can only recover a useful encoder
# for its missing modality via V2V dissemination -- the regime where sharing
# matters. (raymob = full 256-beam task; raymob8 = 8 Tx sectors, clearer signal.)
DATASETS = {
    "raymob8": dict(dataset="raymob8", fusion_mode="concat", starve_frac=0.20,
                    phi_agg=0.15, prefix="rm", main_tag="rmmain"),
}

jobs = []
for dkey, dcfg in DATASETS.items():
    pf = dcfg["prefix"]
    mt = dcfg["main_tag"]
    base_extra = dict(dataset=dcfg["dataset"], fusion_mode=dcfg["fusion_mode"],
                      starve_frac=dcfg["starve_frac"], phi_agg=dcfg["phi_agg"])

    # 1) main comparison: 3 seeds, medium density. RECD uses the Markov-kernel
    #    reachability (no_gat): torch_geometric is unavailable in this env, and
    #    the runtime note records it is numerically equivalent to the GAT here.
    for m, s in itertools.product(METHODS, [1, 2, 3]):
        jobs.append(dict(tag=mt, method=m, trace=f"med_n60_s{s}", seed=s,
                         no_gat=True, **base_extra))

    # 2) V2V range sweep (seed 1; R=200 reused from main)
    for m, R in itertools.product(METHODS, [100, 150, 300]):
        jobs.append(dict(tag=f"{pf}sweepR", method=m, trace="med_n60_s1",
                         seed=1, r_v2v=R, no_gat=True, **base_extra))

    # 3) cache capacity sweep (cache-using methods; C=4 reused from main)
    for m, C in itertools.product(CACHE_METHODS, [2, 6, 8]):
        jobs.append(dict(tag=f"{pf}sweepC", method=m, trace="med_n60_s1",
                         seed=1, cache_encoders=C, no_gat=True, **base_extra))

    # 4) background traffic density sweep (med reused from main)
    for m, d in itertools.product(METHODS, ["low", "high"]):
        jobs.append(dict(tag=f"{pf}sweepD", method=m, trace=f"{d}_n60_s1",
                         seed=1, no_gat=True, **base_extra))


def run_job(job):
    cmd = [PY, RUNNER,
           "--method", job["method"],
           "--trace", job["trace"],
           "--seed", str(job["seed"]),
           "--rounds", str(ROUNDS),
           "--tag", job["tag"],
           "--dataset", job["dataset"],
           "--fusion-mode", job["fusion_mode"],
           "--starve-frac", str(job["starve_frac"]),
           "--phi-agg", str(job["phi_agg"])]
    if "r_v2v" in job:
        cmd += ["--r-v2v", str(job["r_v2v"])]
    if "cache_encoders" in job:
        cmd += ["--cache-encoders", str(job["cache_encoders"])]
    if job.get("no_gat"):
        cmd += ["--no-gat"]
    name = "_".join(
        str(job.get(k, "")) for k in
        ["tag", "method", "trace", "r_v2v", "cache_encoders", "seed"]
    ).replace("__", "_")
    log = os.path.join(LOG_DIR, name + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=RUN_ENV)
    print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rc


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None  # optional dataset filter
    if only:
        jobs = [j for j in jobs if j["dataset"] == only]
    print(f"{len(jobs)} jobs, {WORKERS} workers", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rcs = list(ex.map(run_job, jobs))
    print(f"done: {sum(1 for r in rcs if r == 0)}/{len(rcs)} succeeded")
