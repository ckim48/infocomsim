"""Does the paper's learning-gain-gated aggregation fix negative transfer?

Naive size-weighted aggregation made sharing net-negative (good vehicles got
damaged). The paper's G^learn = [.]^+ guard says: only aggregate an encoder
when it reduces the receiver's local loss. We compare, under 20% starvation:

  NoShare                         (no sharing, pure local -> reference)
  DFL-Gossip / RECD  x  {naive, gated aggregation}

If gated aggregation works, the federated (non-starved) vehicles should stop
losing accuracy (acc_fed ~ NoShare) while starved vehicles still gain, making
sharing net-positive.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run                       # noqa: E402
from methods import METHODS, DFLGossip       # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")


class NoShare(DFLGossip):
    name = "NoShare"

    def decide(self, ctx):
        return []


METHODS["NoShare"] = NoShare

SEEDS = [1, 2]
ROUNDS = 25
# (label, method, gated_agg)
CONDS = [
    ("NoShare", "NoShare", False),
    ("DFL-naive", "DFL-Gossip", False),
    ("DFL-gated", "DFL-Gossip", True),
    ("RECD-naive", "RECD", False),
    ("RECD-gated", "RECD", True),
]


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
    for label, method, gated in CONDS:
        rows = []
        for s in SEEDS:
            cfg = {
                "method": method, "trace": TRACE, "seed": s, "rounds": ROUNDS,
                "r_v2v": 200.0, "cache_encoders": 4, "max_out": 3,
                "V": 50.0, "use_gat": False, "dataset": "fmnist",
                "phi_agg": 0.15, "local_steps": 10, "starve_frac": 0.20,
                "gated_agg": gated,
            }
            print(f"\n=== {label} seed={s} ===", flush=True)
            rows.append(summarize(run(cfg)))
        results[label] = {k: float(np.mean([r[k] for r in rows]))
                          for k in rows[0]}

    with open(os.path.join(BASE, "results", "diag_gated.json"), "w") as f:
        json.dump(results, f, indent=1)

    print("\n===== gated vs naive aggregation (starve 20%, R=25) =====")
    print(f'{"condition":12}{"acc_all":>9}{"acc_starved":>13}'
          f'{"acc_fed":>9}{"cov":>7}')
    for label, _, _ in CONDS:
        r = results[label]
        print(f'{label:12}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}'
              f'{r["acc_fed"]:9.1f}{r["coverage_end"]:7.2f}')
    print("\nsaved results/diag_gated.json")


if __name__ == "__main__":
    main()
