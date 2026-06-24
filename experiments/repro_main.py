"""Full reproduction for paper Sec. IV.

RECD (proposed) vs DFL-Gossip / LRU-Random / Mobility-Greedy on the UCI-HAR
multimodal testbed (complementary body_acc / body_gyro / total_acc sensors,
concat fusion), with starve_frac=0.20 modality starvation. Sweeps vehicle
density (low / med / high) over 3 seeds and records both the final-round
breakdown (all vs starved accuracy, fairness, latency, coverage) and the
per-round convergence curve.

RECD uses the analytic Markov per-segment reachability for Gamma_j (use_gat
off): equivalent to the hierarchical GAT predictor in our ablations but ~100x
faster, which keeps the full 3x4x3 matrix tractable.

Run:  ~/anaconda3/envs/cm-pfl/bin/python experiments/repro_main.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(BASE, "results", "traces")

DENSITIES = ["low", "med", "high"]          # trace_<d>_n60_s1.npz
METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
SEEDS = [1, 2, 3]
ROUNDS = 25


def starved_mask(h):
    """A vehicle is 'starved' if its smallest modality dataset is <= 5."""
    return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])


def summarize(h):
    a = np.array(h["acc_per_veh"]) * 100.0
    m = starved_mask(h)
    return dict(
        acc_all=float(a.mean()),
        acc_starved=float(a[m].mean()) if m.any() else float("nan"),
        n_starved=int(m.sum()),
        jain=float(h.get("jain")),
        mfr=float(h.get("mean_first_recv")),
        cov=float(h["coverage"][-1]),
        energy=float(np.sum(h["cum_energy"])),
        acc_curve=list(h["acc"]),
        acc_rounds=list(h["acc_round"]),
    )


def mean_over_seeds(rows, key):
    vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def sd_over_seeds(rows, key):
    vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def main():
    t0 = time.time()
    results = {}        # "<method>|<density>" -> aggregated dict
    raw = {}            # same key -> list of per-seed summaries (with curves)
    for d in DENSITIES:
        trace = os.path.join(TRACE_DIR, f"trace_{d}_n60_s1.npz")
        if not os.path.exists(trace):
            print(f"!! missing trace {trace}, skipping density {d}", flush=True)
            continue
        for m in METHODS:
            rows = []
            for s in SEEDS:
                cfg = dict(method=m, trace=trace, seed=s, rounds=ROUNDS,
                           r_v2v=200.0, cache_encoders=4, max_out=3, V=50.0,
                           use_gat=False, dataset="har", phi_agg=0.15,
                           local_steps=10, starve_frac=0.20,
                           gated_agg=False, fusion_mode="concat")
                print(f"\n=== density={d} {m} seed={s} ===", flush=True)
                rows.append(summarize(run(cfg)))
            key = f"{m}|{d}"
            raw[key] = rows
            results[key] = {
                "acc_all": mean_over_seeds(rows, "acc_all"),
                "acc_all_sd": sd_over_seeds(rows, "acc_all"),
                "acc_starved": mean_over_seeds(rows, "acc_starved"),
                "acc_starved_sd": sd_over_seeds(rows, "acc_starved"),
                "n_starved": rows[0]["n_starved"],
                "jain": mean_over_seeds(rows, "jain"),
                "mfr": mean_over_seeds(rows, "mfr"),
                "cov": mean_over_seeds(rows, "cov"),
                "energy": mean_over_seeds(rows, "energy"),
            }

    out = {"results": results, "raw": raw,
           "meta": dict(densities=DENSITIES, methods=METHODS, seeds=SEEDS,
                        rounds=ROUNDS, dataset="har", starve_frac=0.20)}
    json.dump(out, open(os.path.join(BASE, "results", "repro_main.json"), "w"))

    # ---------------- print tables
    for d in DENSITIES:
        if f"{METHODS[0]}|{d}" not in results:
            continue
        print(f"\n===== HAR starve20%, density={d}, R=200m, {ROUNDS} rounds, "
              f"3 seeds =====")
        print(f'{"method":16}{"acc_all":>9}{"acc_starved":>13}{"±sd":>6}'
              f'{"jain":>7}{"firstRecv":>11}{"cov":>7}')
        for m in METHODS:
            r = results.get(f"{m}|{d}")
            if r is None:
                continue
            print(f'{m:16}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}'
                  f'{r["acc_starved_sd"]:6.1f}{r["jain"]:7.3f}'
                  f'{r["mfr"]:11.2f}{r["cov"]:7.2f}')

    print(f"\nsaved results/repro_main.json  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
