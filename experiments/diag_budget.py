"""Routing value under a contact budget (HAR, complementary modalities).

Reception helps (proven). Now: does road-aware/need-aware ROUTING matter? Only
when not everyone can be served -> tighten the contact budget via R^V2V and see
if RECD's need-aware prioritization beats DFL/LRU/Mobility for starved cars and
fairness. Loose R (200) should tie (coverage saturates); tight R should separate.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
SEEDS = [1, 2]
RVALS = [200.0, 100.0]
def smask(h): return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])
def summ(h):
    a = np.array(h["acc_per_veh"]) * 100; m = smask(h)
    return dict(acc_all=float(a.mean()),
                acc_starved=float(a[m].mean()) if m.any() else None,
                jain=float(h.get("jain")),
                mfr=float(h.get("mean_first_recv")),
                cov=float(h["coverage"][-1]))
res = {}
for R in RVALS:
    for m in METHODS:
        rows = []
        for s in SEEDS:
            cfg = dict(method=m, trace=TRACE, seed=s, rounds=25, r_v2v=R,
                       cache_encoders=4, max_out=3, V=50.0, use_gat=False,
                       dataset="har", phi_agg=0.15, local_steps=10,
                       starve_frac=0.20, gated_agg=False, fusion_mode="concat")
            print(f"\n=== R={R} {m} seed={s} ===", flush=True)
            rows.append(summ(run(cfg)))
        res[f"{m}|R{int(R)}"] = {k: (float(np.mean([r[k] for r in rows]))
                                     if rows[0][k] is not None else None)
                                 for k in rows[0]}
json.dump(res, open(os.path.join(BASE, "results", "diag_budget.json"), "w"), indent=1)
for R in RVALS:
    print(f"\n===== HAR starve20%, R^V2V={int(R)} m, R=25 rounds =====")
    print(f'{"method":16}{"acc_all":>9}{"acc_starved":>13}{"jain":>7}{"firstRecv":>10}{"cov":>7}')
    for m in METHODS:
        r = res[f"{m}|R{int(R)}"]
        print(f'{m:16}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}{r["jain"]:7.3f}'
              f'{r["mfr"]:10.2f}{r["cov"]:7.2f}')
print("\nsaved results/diag_budget.json")
