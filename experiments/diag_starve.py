"""Diagnostic: does data starvation + a tight round budget make the
dissemination methods separate on accuracy (esp. for starved vehicles)?

Compares the 4 methods under {no starvation, 20% starved} at R=15 rounds.
Uses the analytical Markov kernel (use_gat=False) for speed; road-reachability
is still active. Reports overall accuracy, starved-vehicle accuracy, and Jain
fairness so we can see where smart, need-aware routing pays off.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
SEEDS = [1, 2]
ROUNDS = 15


def starved_mask(hist):
    """A vehicle is starved if it owns a modality with <=5 samples."""
    return np.array([min(v["D"].values()) <= 5 for v in hist["veh_meta"]])


def summarize(hist):
    acc = np.array(hist["acc_per_veh"], dtype=float) * 100
    mask = starved_mask(hist)
    return {
        "acc_all": float(acc.mean()),
        "acc_starved": float(acc[mask].mean()) if mask.any() else None,
        "acc_fed": float(acc[~mask].mean()) if (~mask).any() else None,
        "n_starved": int(mask.sum()),
        "jain": float(hist.get("jain")),
        "coverage_end": float(hist["coverage"][-1]),
    }


def main():
    results = {}
    for starve in (0.0, 0.20):
        for m in METHODS:
            rows = []
            for s in SEEDS:
                cfg = {
                    "method": m, "trace": TRACE, "seed": s, "rounds": ROUNDS,
                    "r_v2v": 200.0, "cache_encoders": 4, "max_out": 3,
                    "V": 50.0, "use_gat": False, "dataset": "fmnist",
                    "phi_agg": 0.15, "local_steps": 10, "starve_frac": starve,
                }
                print(f"\n=== starve={starve} {m} seed={s} ===", flush=True)
                rows.append(summarize(run(cfg)))
            agg = {k: (float(np.mean([r[k] for r in rows]))
                       if rows[0][k] is not None else None)
                   for k in rows[0]}
            results[f"{m}|starve{starve}"] = agg

    out = os.path.join(BASE, "results", "diag_starve.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    # pretty table
    for starve in (0.0, 0.20):
        print(f"\n========== starve_frac = {starve} ==========")
        print(f'{"method":16}{"acc_all":>9}{"acc_starved":>13}'
              f'{"acc_fed":>9}{"jain":>7}{"cov":>6}{"  n_starved"}')
        for m in METHODS:
            r = results[f"{m}|starve{starve}"]
            st = f'{r["acc_starved"]:.1f}' if r["acc_starved"] is not None \
                else "  -"
            print(f'{m:16}{r["acc_all"]:9.1f}{st:>13}{r["acc_fed"]:9.1f}'
                  f'{r["jain"]:7.3f}{r["coverage_end"]:6.2f}   {r["n_starved"]}')
    print("\nsaved results/diag_starve.json")


if __name__ == "__main__":
    main()
