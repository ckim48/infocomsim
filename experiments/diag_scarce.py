"""Genuine scarcity: short deadline + max_out=1 so coverage cannot saturate.

Earlier R-sweep never bound (coverage ~0.97 even at R=60). Here rounds=10 and
max_out=1 force a real delivery shortage -> who gets served depends on routing.
HAR, 20% starvation, naive concat fusion, 3 seeds. Expect RECD (need-aware) to
beat DFL/LRU on starved-car accuracy and fairness.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
SEEDS = [1, 2, 3]
def smask(h): return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])
def summ(h):
    a = np.array(h["acc_per_veh"]) * 100; m = smask(h)
    return dict(acc_all=float(a.mean()),
                acc_starved=float(a[m].mean()) if m.any() else None,
                jain=float(h.get("jain")), mfr=float(h.get("mean_first_recv")),
                cov=float(h["coverage"][-1]))
res = {}
for m in METHODS:
    rows = []
    for s in SEEDS:
        cfg = dict(method=m, trace=TRACE, seed=s, rounds=10, r_v2v=150.0,
                   cache_encoders=4, max_out=1, V=50.0, use_gat=False,
                   dataset="har", phi_agg=0.15, local_steps=10,
                   starve_frac=0.20, gated_agg=False, fusion_mode="concat")
        print(f"\n=== {m} seed={s} ===", flush=True)
        rows.append(summ(run(cfg)))
    res[m] = {k: (float(np.mean([r[k] for r in rows]))
                  if rows[0][k] is not None else None) for k in rows[0]}
    res[m+"|sd"] = {"acc_starved_sd": float(np.std([r["acc_starved"] for r in rows]))}
json.dump(res, open(os.path.join(BASE, "results", "diag_scarce.json"), "w"), indent=1)
print("\n===== HAR starve20%, SCARCE (rounds=10, max_out=1, R=150), 3 seeds =====")
print(f'{"method":16}{"acc_all":>9}{"acc_starved":>13}{"±sd":>6}{"jain":>7}{"firstRecv":>10}{"cov":>7}')
for m in METHODS:
    r = res[m]; sd = res[m+"|sd"]["acc_starved_sd"]
    print(f'{m:16}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}{sd:6.1f}{r["jain"]:7.3f}'
          f'{r["mfr"]:10.2f}{r["cov"]:7.2f}')
print("\nsaved results/diag_scarce.json")
