"""Decisive sharing test on UCI HAR (genuinely complementary sensor modalities).

HAR fusion(3 modalities)=69% vs single=43% (+26pp), so a vehicle with a poor
encoder for one owned modality has real headroom to gain from receiving a better
one. Test: NoShare vs sharing (naive/gated) under 20% starvation.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
from methods import METHODS, DFLGossip
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
class NoShare(DFLGossip):
    name = "NoShare"
    def decide(self, ctx): return []
METHODS["NoShare"] = NoShare
SEEDS=[1,2]; ROUNDS=25
CONDS=[("NoShare","NoShare",False),("DFL-naive","DFL-Gossip",False),
       ("DFL-gated","DFL-Gossip",True),("RECD-naive","RECD",False),
       ("RECD-gated","RECD",True)]
def smask(h): return np.array([min(v["D"].values())<=5 for v in h["veh_meta"]])
def summ(h):
    a=np.array(h["acc_per_veh"])*100; m=smask(h)
    return dict(acc_all=float(a.mean()),
                acc_starved=float(a[m].mean()) if m.any() else None,
                acc_fed=float(a[~m].mean()) if (~m).any() else None,
                coverage_end=float(h["coverage"][-1]))
res={}
for label,method,gated in CONDS:
    rows=[]
    for s in SEEDS:
        cfg=dict(method=method,trace=TRACE,seed=s,rounds=ROUNDS,r_v2v=200.0,
                 cache_encoders=4,max_out=3,V=50.0,use_gat=False,dataset="har",
                 phi_agg=0.15,local_steps=10,starve_frac=0.20,gated_agg=gated,
                 fusion_mode="concat")
        print(f"\n=== {label} seed={s} ===",flush=True); rows.append(summ(run(cfg)))
    res[label]={k:(float(np.mean([r[k] for r in rows])) if rows[0][k] is not None else None) for k in rows[0]}
json.dump(res,open(os.path.join(BASE,"results","diag_har.json"),"w"),indent=1)
print("\n===== HAR + starve20% + concat fusion, R=25 =====")
print(f'{"cond":12}{"acc_all":>9}{"acc_starved":>13}{"acc_fed":>9}{"cov":>7}')
for label,_,_ in CONDS:
    r=res[label]; st=f'{r["acc_starved"]:.1f}' if r["acc_starved"] is not None else "-"
    print(f'{label:12}{r["acc_all"]:9.1f}{st:>13}{r["acc_fed"]:9.1f}{r["coverage_end"]:7.2f}')
print("saved results/diag_har.json")
