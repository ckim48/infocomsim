"""Find a scenario where road-graph reachability (RECD) beats myopic LET
(Mobility-Greedy). Sparse multi-hop regime (R=100, avg degree ~1.45, 36%
isolated) + short deadline so carry-forward relaying is the bottleneck and the
*choice of relay* matters. Sweep RECD's dissemination weight nu and horizon
h_max to activate the Gamma term; compare against Mobility-Greedy.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
SEEDS = [1, 2, 3]
# (label, method, nu, h_max)
CONDS = [
    ("Mobility-Greedy", "Mobility-Greedy", None, None),
    ("RECD nu0.3 h3", "RECD", 0.3, 3),
    ("RECD nu1.5 h3", "RECD", 1.5, 3),
    ("RECD nu3.0 h6", "RECD", 3.0, 6),
]
def smask(h): return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])
def summ(h):
    a = np.array(h["acc_per_veh"]) * 100; m = smask(h)
    return dict(acc_all=float(a.mean()),
                acc_starved=float(a[m].mean()) if m.any() else None,
                jain=float(h.get("jain")), mfr=float(h.get("mean_first_recv")),
                cov=float(h["coverage"][-1]))
res = {}
for label, method, nu, hmax in CONDS:
    rows = []
    for s in SEEDS:
        cfg = dict(method=method, trace=TRACE, seed=s, rounds=15, r_v2v=100.0,
                   cache_encoders=4, max_out=2, V=50.0, use_gat=False,
                   dataset="har", phi_agg=0.15, local_steps=10,
                   starve_frac=0.20, gated_agg=False, fusion_mode="concat")
        if nu is not None: cfg["nu"] = nu
        if hmax is not None: cfg["h_max"] = hmax
        print(f"\n=== {label} seed={s} ===", flush=True)
        rows.append(summ(run(cfg)))
    res[label] = {k: (float(np.mean([r[k] for r in rows]))
                      if rows[0][k] is not None else None) for k in rows[0]}
    res[label+"|sd"] = float(np.std([r["acc_starved"] for r in rows]))
json.dump(res, open(os.path.join(BASE, "results", "diag_roadaware.json"), "w"), indent=1)
print("\n===== HAR sparse (R=100, rounds=15, max_out=2), starve20%, 3 seeds =====")
print(f'{"condition":18}{"acc_all":>9}{"acc_starved":>13}{"±sd":>6}{"jain":>7}{"firstRecv":>10}{"cov":>7}')
for label, _, _, _ in CONDS:
    r = res[label]
    print(f'{label:18}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}{res[label+"|sd"]:6.1f}'
          f'{r["jain"]:7.3f}{r["mfr"]:10.2f}{r["cov"]:7.2f}')
print("\nsaved results/diag_roadaware.json")
