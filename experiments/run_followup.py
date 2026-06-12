"""Follow-up campaign: (a) re-run mains with per-vehicle metrics,
(b) UCI-HAR main comparison (real multimodal dataset, MFedMC setting)."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import run_job  # noqa: E402
from concurrent.futures import ThreadPoolExecutor

METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]

jobs = []
# (a) re-run fmnist mains (same configs/seeds; engine now records per-vehicle
#     accuracy and heterogeneity metadata) -- overwrites main_*.json
for m, s in itertools.product(METHODS, [1, 2, 3]):
    jobs.append(dict(tag="main", method=m, trace=f"med_n60_s{s}", seed=s))
# (b) UCI-HAR main comparison
for m, s in itertools.product(METHODS, [1, 2, 3]):
    jobs.append(dict(tag="harmain", method=m, trace=f"med_n60_s{s}", seed=s,
                     dataset="har"))

if __name__ == "__main__":
    print(f"{len(jobs)} follow-up jobs", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        rcs = list(ex.map(run_job, jobs))
    print(f"done: {sum(1 for r in rcs if r == 0)}/{len(rcs)} succeeded")
