"""Quick calibration: find a contact-budget regime where coverage < 1.

Sweeps R^V2V, max_out, cache size, density on HAR+starvation with RECD so the
dissemination term has room to matter. Single seed, few rounds -> fast.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = lambda n: os.path.join(BASE, "results", "traces", n)

GRID = [
    dict(tag="low R60 mo1 c2",      trace="trace_low_n60_s1.npz", r=60.0,  mo=1, cache=2),
    dict(tag="low R75 mo1 c2",      trace="trace_low_n60_s1.npz", r=75.0,  mo=1, cache=2),
    dict(tag="med30 R75 mo1 c2",    trace="trace_med_n30_s1.npz", r=75.0,  mo=1, cache=2),
    dict(tag="med60 R60 mo1 c2",    trace="trace_med_n60_s1.npz", r=60.0,  mo=1, cache=2),
    dict(tag="low R50 mo1 c2",      trace="trace_low_n60_s1.npz", r=50.0,  mo=1, cache=2),
]

def smask(h): return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])
for g in GRID:
    cfg = dict(method="RECD", trace=TR(g["trace"]), seed=1, rounds=20,
               r_v2v=g["r"], cache_encoders=g["cache"], max_out=g["mo"],
               V=50.0, nu=0.3, use_gat=False, dataset="har", phi_agg=0.15,
               local_steps=10, starve_frac=0.30, gated_agg=False,
               fusion_mode="concat", eval_every=20)
    h = run(cfg)
    a = np.array(h["acc_per_veh"]) * 100; m = smask(h)
    print(f'{g["tag"]:20} cov={h["coverage"][-1]:.2f} '
          f'mfr={h["mean_first_recv"]:.2f} jain={h["jain"]:.3f} '
          f'starved={a[m].mean():.1f} all={a.mean():.1f} '
          f'succ/round={np.mean(h["succ"]):.1f}', flush=True)
