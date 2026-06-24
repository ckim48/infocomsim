"""Dual-objective hypothesis test.

The paper's contribution is NOT "give a vehicle a learning-useful encoder," but
"give a learning-useful encoder while also caching/relaying it through vehicles
that will further spread it to more *needy* vehicles in the future."

That dual objective is exactly RECD's utility   learn + nu * p_tx * Gamma_j
(methods.RECD.decide / select_cache). We isolate the second term by ablating nu,
holding everything else fixed, in a *budget-constrained* regime (tight V2V range,
max_out=1, small cache) so coverage stays < 100% and routing actually matters.

Conditions (all RECD, identical except nu / road-awareness):
  learn-only   nu=0            -> only immediate learning benefit to the receiver
  +dissem 0.3  nu=0.3          -> + future dissemination potential (road-aware Gamma)
  +dissem 1.0  nu=1.0          -> stronger weight on spread
Reference baselines: DFL-Gossip (no cache/relay), LRU-Random (relay, no routing).

If the hypothesis holds, adding the dissemination term should reach MORE needy
(starved) vehicles, FASTER, raising starved coverage / lowering starved
first-reception time / raising starved accuracy and fairness.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from engine import run
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- budget-constrained regime (set from calib_budget.py) ---
TRACE_NAME = "trace_low_n60_s1.npz"   # may be overridden after calibration
R_V2V      = 60.0
MAX_OUT    = 1
CACHE      = 2
STARVE     = 0.30
ROUNDS     = 25
SEEDS      = [1, 2, 3]

TRACE = os.path.join(BASE, "results", "traces", TRACE_NAME)

# (label, method, nu, need_dissem)   nu/need ignored by non-RECD methods
CONDS = [
    ("DFL-Gossip",      "DFL-Gossip", 0.0, False),
    ("LRU-Random",      "LRU-Random", 0.0, False),
    ("RECD learn-only", "RECD",       0.0, False),  # ν=0: receiver benefit only
    ("RECD +reach.3",   "RECD",       0.3, False),  # +Γ_j reachability (paper)
    ("RECD +reach1.0",  "RECD",       1.0, False),
    ("RECD +need.3",    "RECD",       0.3, True),   # +Γ_j·S_r need-aware (hypo.)
    ("RECD +need1.0",   "RECD",       1.0, True),
]


def smask(h):
    return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])


def summ(h):
    a = np.array(h["acc_per_veh"]) * 100
    m = smask(h)
    starved_ids = set(np.where(m)[0].tolist())
    # need pairs (i,r) for starved vehicles, and which of them got served + when
    need_starved, recv_rounds = 0, []
    for vid in starved_ids:
        for r in h["veh_meta"][vid]["mods"]:
            need_starved += 1
            fr = h["first_recv"].get(f"{vid}_{r}")
            if fr is not None:
                recv_rounds.append(fr)
    starved_cov = len(recv_rounds) / max(need_starved, 1)
    starved_mfr = float(np.mean(recv_rounds)) if recv_rounds else float(ROUNDS)
    return dict(
        acc_all=float(a.mean()),
        acc_starved=float(a[m].mean()) if m.any() else None,
        acc_fed=float(a[~m].mean()) if (~m).any() else None,
        cov=float(h["coverage"][-1]),
        starved_cov=float(starved_cov),
        starved_mfr=float(starved_mfr),
        jain=float(h.get("jain")),
        mfr=float(h.get("mean_first_recv")),
    )


def main():
    res = {}
    for label, method, nu, need in CONDS:
        rows = []
        for s in SEEDS:
            cfg = dict(method=method, trace=TRACE, seed=s, rounds=ROUNDS,
                       r_v2v=R_V2V, cache_encoders=CACHE, max_out=MAX_OUT,
                       V=50.0, nu=nu, need_dissem=need, use_gat=False,
                       dataset="har", phi_agg=0.15, local_steps=10,
                       starve_frac=STARVE, gated_agg=False,
                       fusion_mode="concat", eval_every=ROUNDS)
            print(f"\n=== {label} seed={s} ===", flush=True)
            rows.append(summ(run(cfg)))
        res[label] = {k: (float(np.mean([r[k] for r in rows]))
                          if rows[0][k] is not None else None)
                      for k in rows[0]}
    out = os.path.join(BASE, "results", "diag_dualobj.json")
    json.dump(dict(meta=dict(trace=TRACE_NAME, r_v2v=R_V2V, max_out=MAX_OUT,
                             cache=CACHE, starve=STARVE, rounds=ROUNDS,
                             seeds=SEEDS), res=res),
              open(out, "w"), indent=1)
    print(f"\n===== Dual-objective ablation: HAR starve{int(STARVE*100)}%, "
          f"R={int(R_V2V)}m, max_out={MAX_OUT}, cache={CACHE}, "
          f"{len(SEEDS)} seeds =====")
    hdr = (f'{"condition":17}{"acc_all":>8}{"acc_strv":>9}{"acc_fed":>8}'
           f'{"cov":>6}{"strvCov":>8}{"strvFR":>8}{"jain":>6}')
    print(hdr)
    for label, *_ in CONDS:
        r = res[label]
        st = f'{r["acc_starved"]:.1f}' if r["acc_starved"] is not None else "-"
        print(f'{label:17}{r["acc_all"]:8.1f}{st:>9}{r["acc_fed"]:8.1f}'
              f'{r["cov"]:6.2f}{r["starved_cov"]:8.2f}{r["starved_mfr"]:8.2f}'
              f'{r["jain"]:6.3f}')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
