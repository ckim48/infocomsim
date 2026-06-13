"""Foundational test: does RECEIVING a disseminated encoder actually help?

The dissemination/road-aware story is only meaningful if a vehicle that
receives useful encoders learns better than one that trains in isolation.
Here we add the missing control group: NoShare (no V2V transmission at all,
pure local training) and compare it against full sharing (DFL-Gossip, which
reaches ~100% coverage) and RECD, under 20% data starvation.

Decisive metric: starved-vehicle accuracy. If sharing >> NoShare for starved
vehicles, reception works and dissemination is justified. If sharing ~= NoShare
even at full coverage, the benefit channel is broken (receiving doesn't help).
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run            # noqa: E402
from methods import METHODS, DFLGossip  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")


class NoShare(DFLGossip):
    """Control: never transmit -> every vehicle trains only on its own data."""
    name = "NoShare"

    def decide(self, ctx):
        return []


METHODS["NoShare"] = NoShare

CONDITIONS = ["NoShare", "DFL-Gossip", "RECD"]
SEEDS = [1, 2]
ROUNDS = 25


def starved_mask(hist):
    return np.array([min(v["D"].values()) <= 5 for v in hist["veh_meta"]])


def summarize(hist):
    acc = np.array(hist["acc_per_veh"], dtype=float) * 100
    mask = starved_mask(hist)
    return {
        "acc_all": float(acc.mean()),
        "acc_starved": float(acc[mask].mean()) if mask.any() else None,
        "acc_fed": float(acc[~mask].mean()) if (~mask).any() else None,
        "coverage_end": float(hist["coverage"][-1]),
    }


def main():
    results = {}
    for m in CONDITIONS:
        rows = []
        for s in SEEDS:
            cfg = {
                "method": m, "trace": TRACE, "seed": s, "rounds": ROUNDS,
                "r_v2v": 200.0, "cache_encoders": 4, "max_out": 3,
                "V": 50.0, "use_gat": False, "dataset": "fmnist",
                "phi_agg": 0.15, "local_steps": 10, "starve_frac": 0.20,
            }
            print(f"\n=== {m} seed={s} ===", flush=True)
            rows.append(summarize(run(cfg)))
        results[m] = {k: (float(np.mean([r[k] for r in rows]))
                          if rows[0][k] is not None else None)
                      for k in rows[0]}

    with open(os.path.join(BASE, "results", "diag_reception.json"), "w") as f:
        json.dump(results, f, indent=1)

    print("\n========== reception benefit (starve 20%, R=25) ==========")
    print(f'{"condition":14}{"acc_all":>9}{"acc_starved":>13}'
          f'{"acc_fed":>9}{"cov":>7}')
    base = results["NoShare"]
    for m in CONDITIONS:
        r = results[m]
        print(f'{m:14}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}'
              f'{r["acc_fed"]:9.1f}{r["coverage_end"]:7.2f}')
    lift = results["DFL-Gossip"]["acc_starved"] - base["acc_starved"]
    print(f'\nstarved-vehicle lift from sharing (DFL - NoShare): {lift:+.1f} pp')
    print("saved results/diag_reception.json")


if __name__ == "__main__":
    main()
