"""Geographic starvation + tight budget: the regime where mobility prediction
should matter. Modality 0 is starved on LEFT-half vehicles (demand on the
left); good modality-0 encoders live on the RIGHT, so they must be CARRIED
across the map. With a tight contact budget (coverage < 100%), the choice of
relay -- whether a vehicle heads toward the needy left region -- determines who
gets served. Here vehicle-specific reachability (HierGAT) should beat the
shared kernel and Gamma-off.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")

P0 = np.load(TRACE, allow_pickle=True)["pos"][:, 0, :]
xmed = np.median(P0[:, 0])
LEFT = [int(i) for i in range(P0.shape[0]) if P0[i, 0] < xmed]   # needy region
print(f"left/needy region: {len(LEFT)} vehicles (x < {xmed:.0f})", flush=True)

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
                n_starved=int(m.sum()),
                jain=float(h.get("jain")), mfr=float(h.get("mean_first_recv")),
                cov=float(h["coverage"][-1]))
res = {}
for label, method, ug, nu, hmax in CONDS:
    rows = []
    for s in SEEDS:
        cfg = dict(method=method, trace=TRACE, seed=s, rounds=12, r_v2v=100.0,
                   cache_encoders=4, max_out=1, V=50.0, use_gat=ug,
                   dataset="har", phi_agg=0.15, local_steps=10,
                   starve_ids=LEFT, starve_mod=0,
                   gated_agg=False, fusion_mode="concat",
                   gat_epochs=150, gat_n_t=80)
        if nu is not None: cfg["nu"] = nu
        if hmax is not None: cfg["h_max"] = hmax
        print(f"\n=== {label} seed={s} ===", flush=True)
        rows.append(summ(run(cfg)))
    res[label] = {k: (float(np.mean([r[k] for r in rows]))
                      if rows[0][k] is not None else None) for k in rows[0]}
    res[label+"|sd"] = float(np.std([r["acc_starved"] for r in rows]))
json.dump(res, open(os.path.join(BASE, "results", "diag_geo.json"), "w"), indent=1)
print("\n===== Geographic starvation (left mod-0) + tight budget, HAR, 3 seeds =====")
print(f'{"condition":24}{"acc_all":>9}{"acc_starved":>13}{"±sd":>6}{"jain":>7}{"firstRecv":>10}{"cov":>7}')
for label, _, _, _, _ in CONDS:
    r = res[label]
    print(f'{label:24}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}{res[label+"|sd"]:6.1f}'
          f'{r["jain"]:7.3f}{r["mfr"]:10.2f}{r["cov"]:7.2f}')
print(f'(n_starved per run ~ {res["RECD HierGAT (veh)"]["n_starved"]})')
print("\nsaved results/diag_geo.json")
