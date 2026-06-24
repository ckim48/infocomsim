"""Summarize the encoder-level study (tag=enc).

Per method, pooled over seeds: per-modality ENCODER probe accuracy split into
STARVED vs FED (vehicle, modality) pairs. The paper's premise -- sharing a
useful encoder helps a vehicle that lacks that modality's data -- predicts that
the sharing methods beat Local on STARVED-modality encoder accuracy.
"""
import glob
import json
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "results", "runs")
METHODS = ["Local", "RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
MODNAME = {0: "GPS", 1: "image", 2: "lidar"}
TAG = sys.argv[1] if len(sys.argv) > 1 else "enc"   # "enc" or "encg" (gated)


def runs_for(method):
    out = []
    for f in glob.glob(os.path.join(RUNS, f"{TAG}_{method}_*.json")):
        out.append(json.load(open(f))["hist"])
    return out


def pooled(method):
    """Return dict (mod, grp)->list of encoder accs pooled over seeds, plus
    overall fused acc / coverage / energy lists."""
    agg = {}
    fused, cov, energy = [], [], []
    for h in runs_for(method):
        if "enc_acc_per_veh" not in h:
            continue
        fused.append(h["acc"][-1] * 100)
        cov.append(h["coverage"][-1] * 100)
        energy.append(np.sum(h["energy"]) / 1e3)
        for e, m in zip(h["enc_acc_per_veh"], h["veh_meta"]):
            for r, acc in e.items():
                r = int(r)
                grp = "starved" if m["D"][str(r)] <= 5 else "fed"
                agg.setdefault((r, grp), []).append(acc * 100)
    return agg, fused, cov, energy


if __name__ == "__main__":
    print("Encoder-level study  (Raymobtime 256-beam, real SF GPS, K=100, "
          "starve 30%)\nENCODER probe accuracy %% (chance 0.4%); "
          "STARVED = vehicle lacks that modality's data\n")
    # header
    cols = [(r, g) for r in (0, 1, 2) for g in ("starved", "fed")]
    hdr = f"{'method':16s}" + "".join(
        f"{MODNAME[r][:4]+'/'+g[:4]:>12s}" for r, g in cols) \
        + f"{'fused%':>8s}{'cov%':>7s}{'en.kJ':>7s}"
    print(hdr)
    base = {}
    for m in METHODS:
        agg, fused, cov, energy = pooled(m)
        if not fused:
            continue
        row = f"{m:16s}"
        for r, g in cols:
            v = agg.get((r, g), [])
            row += f"{(np.mean(v) if v else float('nan')):8.1f}({len(v):2d})"
        row += f"{np.mean(fused):8.1f}{np.mean(cov):7.1f}{np.mean(energy):7.1f}"
        print(row)
        base[m] = agg
    # key contrast: sharing vs Local on STARVED modality encoders
    if "Local" in base:
        print("\nSTARVED-modality encoder gain vs Local (pp):")
        for r in (0, 1, 2):
            loc = base["Local"].get((r, "starved"), [])
            if not loc:
                continue
            lm = np.mean(loc)
            deltas = []
            for m in METHODS:
                if m == "Local":
                    continue
                v = base.get(m, {}).get((r, "starved"), [])
                deltas.append(f"{m.split('-')[0]}:{np.mean(v)-lm:+.1f}" if v else f"{m}:NA")
            print(f"  {MODNAME[r]:6s} (Local={lm:.1f}%, n={len(loc)}): " + "  ".join(deltas))
