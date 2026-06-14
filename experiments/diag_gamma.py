"""Clean Gamma ablation: does road-aware reachability help, all else equal?

Within RECD, nu=0 turns OFF road-awareness in BOTH stages -- caching value
becomes pure learning need (no reachability g) and dissemination drops the
nu*ptx*Gamma term, so relaying to non-owner carriers stops (w<=0 for them).
nu>0 turns it ON. Everything else (contact prediction, Y/Z queues, gating) is
identical. Sparse multi-hop regime where carry-forward relaying is the
bottleneck. Mobility-Greedy kept as an external reference.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
SEEDS = [1, 2, 3]
# (label, method, nu, h_max)
CONDS = [
    ("Mobility-Greedy (ref)", "Mobility-Greedy", None, None),
    ("RECD Gamma OFF (nu0)", "RECD", 0.0, 3),
    ("RECD Gamma ON nu1.5 h6", "RECD", 1.5, 6),
    ("RECD Gamma ON nu3.0 h6", "RECD", 3.0, 6),
]
def smask(h): return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])
def summ(h):
    a = np.array(h["acc_per_veh"]) * 100; m = smask(h)
    return dict(acc_all=float(a.mean()),
                acc_starved=float(a[m].mean()) if m.any() else None,
                jain=float(h.get("jain")), mfr=float(h.get("mean_first_recv")),
                recv=float(np.sum(h["recv_per_veh"])), cov=float(h["coverage"][-1]))
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
json.dump(res, open(os.path.join(BASE, "results", "diag_gamma.json"), "w"), indent=1)
print("\n===== Gamma on/off ablation (HAR sparse R=100, rounds=15, starve20%, 3 seeds) =====")
print(f'{"condition":26}{"acc_all":>9}{"acc_starved":>13}{"±sd":>6}{"jain":>7}{"firstRecv":>10}{"#recv":>8}{"cov":>7}')
for label, _, _, _ in CONDS:
    r = res[label]
    print(f'{label:26}{r["acc_all"]:9.1f}{r["acc_starved"]:13.1f}{res[label+"|sd"]:6.1f}'
          f'{r["jain"]:7.3f}{r["mfr"]:10.2f}{r["recv"]:8.0f}{r["cov"]:7.2f}')
print("\nsaved results/diag_gamma.json")
