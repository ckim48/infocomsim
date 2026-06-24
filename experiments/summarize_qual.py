"""Summarize the harsh-quality study (tag=encq): does sharing a clean encoder
help vehicles with a degraded (noisy) sensor? Pooled over seeds: per-method
low-quality-modality ENCODER accuracy and low-quality-vehicle FUSED accuracy,
with the gain over the Local (no-sharing) control.
"""
import glob
import json
import os
import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "runs")
METHODS = ["Local", "RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
TAG = "encq"


def runs_for(m):
    return [json.load(open(f))["hist"]
            for f in glob.glob(os.path.join(RUNS, f"{TAG}_{m}_*.json"))]


def stats(m):
    le, he, lf, hf, en = ([] for _ in range(5))
    for h in runs_for(m):
        vm = h["veh_meta"]
        acc = np.array(h["acc_per_veh"])
        lowq_veh = np.array([any(q < 0.6 for q in d["Q"].values()) for d in vm])
        if lowq_veh.any():
            lf.append(acc[lowq_veh].mean() * 100)
        hf.append(acc[~lowq_veh].mean() * 100)
        en.append(np.sum(h["energy"]) / 1e3)
        if "enc_acc_per_veh" in h:
            a, b = [], []
            for e, d in zip(h["enc_acc_per_veh"], vm):
                for r, x in e.items():
                    (a if d["Q"][str(r)] < 0.6 else b).append(x * 100)
            le.append(np.mean(a)); he.append(np.mean(b))
    mn = lambda x: float(np.mean(x)) if len(x) else float("nan")
    sd = lambda x: float(np.std(x)) if len(x) else 0.0
    return dict(le=mn(le), les=sd(le), he=mn(he), lf=mn(lf), lfs=sd(lf),
               hf=mn(hf), en=mn(en), n=len(hf))


if __name__ == "__main__":
    print("Harsh-quality regime (complementary FashionMNIST, real SF GPS, K=100, "
          "full data, ungated)\nlow-quality = degraded sensor (noisy, full data)\n")
    print(f"{'method':16s}{'lowQ ENC':>14s}{'hiQ ENC':>9s}"
          f"{'lowQ-veh FUSED':>16s}{'hiQ FUSED':>10s}{'en.kJ':>7s}")
    base = None
    for m in METHODS:
        s = stats(m)
        if not s["n"]:
            continue
        print(f"{m:16s}{s['le']:8.1f}±{s['les']:3.1f}{s['he']:9.1f}"
              f"{s['lf']:11.1f}±{s['lfs']:3.1f}{s['hf']:10.1f}{s['en']:7.1f}")
        if m == "Local":
            base = s
    if base:
        print("\nGain vs Local (no sharing):")
        for m in METHODS:
            if m == "Local":
                continue
            s = stats(m)
            if s["n"]:
                print(f"  {m:16s} lowQ ENC {s['le']-base['le']:+5.1f}pp   "
                      f"lowQ-veh FUSED {s['lf']-base['lf']:+5.1f}pp")
