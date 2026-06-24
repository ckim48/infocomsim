"""Does the hierarchical GAT (vehicle-specific reachability) help RECD?

Compares, all else equal (HAR, sparse R=100, rounds=15, max_out=2, starve20%):
  RECD Gamma-off (nu=0)            -- no road-awareness
  RECD Markov per-segment (no_gat) -- shared analytical kernel
  RECD HierGAT (vehicle-specific)  -- the proposed predictor
Mobility-Greedy kept as external reference. HierGAT trained with enough epochs
and sampled timesteps.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
SEEDS = [1, 2, 3]
# (label, method, use_gat, nu, h_max)
CONDS = [
    ("Mobility-Greedy (ref)", "Mobility-Greedy", False, None, None),
    ("RECD Gamma-off (nu0)",  "RECD", False, 0.0, 3),
    ("RECD Markov per-seg",   "RECD", False, 2.0, 4),
    ("RECD HierGAT (veh)",    "RECD", True,  2.0, 4),
]
def smask(h): return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])
def summ(h):
    a = np.array(h["acc_per_veh"]) * 100; m = smask(h)
    return dict(acc_all=float(a.mean()),
                acc_starved=float(a[m].mean()) if m.any() else None,
                jain=float(h.get("jain")), mfr=float(h.get("mean_first_recv")),
                cov=float(h["coverage"][-1]), wall=float(h.get("wall_sec", 0)))
res = {}
for label, method, ug, nu, hmax in CONDS:
    rows = []
    for s in SEEDS:
        cfg = dict(method=method, trace=TRACE, seed=s, rounds=15, r_v2v=100.0,
                   cache_encoders=4, max_out=2, V=50.0, use_gat=ug,
                   dataset="har", phi_agg=0.15, local_steps=10,
                   starve_frac=0.20, gated_agg=False, fusion_mode="concat",
                   gat_epochs=150, gat_n_t=80)
        if nu is not None: cfg["nu"] = nu
        if hmax is not None: cfg["h_max"] = hmax
        print(f"\n=== {label} seed={s} ===", flush=True)
        rows.append(summ(run(cfg)))
    res[label] = {k: (float(np.mean([r[k] for r in rows]))
                      if rows[0][k] is not None else None) for k in rows[0]}
    res[label+"|sd"] = float(np.std([r["acc_starved"] for r in rows]))
json.dump(res, open(os.path.join(BASE, "results", "diag_hiergat.json"), "w"), indent=1)
print("\n===== HierGAT vs Markov vs Gamma-off (HAR sparse, starve20%, 3 seeds) =====")
print(f'{"condition":24}{"acc_all":>9}{"acc_starved":>13}{"±sd":>6}{"jain":>7}{"firstRecv":>10}{"cov":>7}')
for label, _, _, _, _ in CONDS:
    r = res[label]
    print(f'{label:24}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}{res[label+"|sd"]:6.1f}'
          f'{r["jain"]:7.3f}{r["mfr"]:10.2f}{r["cov"]:7.2f}')
print("\nsaved results/diag_hiergat.json")
