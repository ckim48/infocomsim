"""Summarize the complementary-FashionMNIST study (tag=encc): does sharing help
when the encoder is the bottleneck? Reports, pooled over seeds, the FUSED
accuracy of starved vs fed vehicles and the per-modality ENCODER accuracy of
starved vs fed (vehicle, modality) pairs, per method.
"""
import glob
import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "results", "runs")
METHODS = ["Local", "RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
TAG = "encc"


def runs_for(m):
    return [json.load(open(f))["hist"]
            for f in glob.glob(os.path.join(RUNS, f"{TAG}_{m}_*.json"))]


def stats(m):
    sf, ff, se, fe, ov, cov, en = ([] for _ in range(7))
    for h in runs_for(m):
        acc = np.array(h["acc_per_veh"])
        vm = h["veh_meta"]
        st = np.array([min(d["D"].values()) <= 5 for d in vm])
        if st.any():
            sf.append(acc[st].mean() * 100)
        ff.append(acc[~st].mean() * 100)
        ov.append(h["acc"][-1] * 100)
        cov.append(h["coverage"][-1] * 100)
        en.append(np.sum(h["energy"]) / 1e3)
        if "enc_acc_per_veh" in h:
            s, f = [], []
            for e, d in zip(h["enc_acc_per_veh"], vm):
                for r, a in e.items():
                    (s if d["D"][str(r)] <= 5 else f).append(a * 100)
            se.append(np.mean(s)); fe.append(np.mean(f))
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return dict(sf=mean(sf), ff=mean(ff), se=mean(se), fe=mean(fe),
               ov=mean(ov), cov=mean(cov), en=mean(en), n=len(ov))


if __name__ == "__main__":
    print("Complementary FashionMNIST (encoder is the bottleneck), real SF GPS, "
          "K=100, gated, starve 30%\n")
    print(f"{'method':16s}{'FUSED starv':>12s}{'FUSED fed':>10s}"
          f"{'ENC starv':>10s}{'ENC fed':>9s}{'overall':>8s}{'cov%':>7s}{'en.kJ':>7s}")
    base = None
    for m in METHODS:
        s = stats(m)
        if not s["n"]:
            continue
        print(f"{m:16s}{s['sf']:12.1f}{s['ff']:10.1f}{s['se']:10.1f}"
              f"{s['fe']:9.1f}{s['ov']:8.1f}{s['cov']:7.1f}{s['en']:7.1f}")
        if m == "Local":
            base = s
    if base:
        print(f"\nStarved-vehicle gain vs Local (no sharing):")
        for m in METHODS:
            if m == "Local":
                continue
            s = stats(m)
            if not s["n"]:
                continue
            print(f"  {m:16s} FUSED {s['sf']-base['sf']:+5.1f}pp   "
                  f"ENC {s['se']-base['se']:+5.1f}pp")
