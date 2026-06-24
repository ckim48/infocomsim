"""Summary tables for the paper campaign (both testbeds).

For each dataset (FashionMNIST: tag=main; UCI HAR: tag=harmain) prints a
method-comparison table over the 3 seeds, plus the breakdown that actually
tests the paper's premise: do STARVED vehicles (one owned modality reduced to
~3 samples) end up more accurate with dissemination than without?  A starved
vehicle can only recover a usable encoder for that modality via V2V, so this
is where RECD must beat the no-road-awareness baselines if the claim holds.

Run:  ~/anaconda3/envs/cm-pfl/bin/python experiments/summarize_paper.py
"""
import glob
import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "results", "runs")
METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
CACHE_METHODS = ["RECD", "LRU-Random", "Mobility-Greedy"]


def load(tag=None, method=None, trace=None, R=None, C=None, seed=None):
    out = []
    for f in glob.glob(os.path.join(RUNS, "*.json")):
        parts = os.path.basename(f)[:-5].split("_")
        d = dict(tag=parts[0], method=parts[1])
        try:
            d["R"] = int([p for p in parts if p.startswith("R")][-1][1:])
            d["C"] = int([p for p in parts if p.startswith("C")][-1][1:])
            d["seed"] = int(parts[-1][1:])
            d["trace"] = "_".join(parts[2:-4])
        except (ValueError, IndexError):
            continue
        if ((tag is None or d["tag"] == tag) and (method is None or d["method"] == method)
                and (trace is None or d["trace"] == trace) and (R is None or d["R"] == R)
                and (C is None or d["C"] == C) and (seed is None or d["seed"] == seed)):
            with open(f) as fh:
                j = json.load(fh)
            j["meta"] = d
            out.append(j)
    return out


def starved_mask(h):
    """Vehicles whose smallest owned-modality dataset is tiny (starved)."""
    return np.array([min(v["D"].values()) <= 5 for v in h["veh_meta"]])


def agg(runs, fn):
    vals = [fn(r["hist"]) for r in runs]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def main_table(tag, title):
    print(f"\n{'='*78}\n{title}  (tag={tag})\n{'='*78}")
    hdr = (f"{'method':16s}{'acc%':>11s}{'starved%':>11s}{'fed%':>9s}"
           f"{'cov%':>7s}{'succ%':>7s}{'energy.kJ':>10s}{'1stRecv':>8s}")
    print(hdr)
    base_starved = None
    for m in METHODS:
        runs = load(tag=tag, method=m)
        if not runs:
            continue
        acc_m, acc_s = agg(runs, lambda h: h["acc"][-1] * 100)
        cov, _ = agg(runs, lambda h: h["coverage"][-1] * 100)
        succ, _ = agg(runs, lambda h: 100 * np.sum(h["succ"]) / max(np.sum(h["att"]), 1))
        en, _ = agg(runs, lambda h: np.sum(h["energy"]) / 1e3)
        fr, _ = agg(runs, lambda h: h["mean_first_recv"])

        def sv(h, want_starved):
            a = np.array(h["acc_per_veh"]) * 100
            msk = starved_mask(h)
            sel = msk if want_starved else ~msk
            return float(a[sel].mean()) if sel.any() else float("nan")
        st_m, _ = agg(runs, lambda h: sv(h, True))
        fe_m, _ = agg(runs, lambda h: sv(h, False))
        if m == "DFL-Gossip":
            base_starved = st_m
        print(f"{m:16s}{acc_m:7.1f}±{acc_s:3.1f}{st_m:11.1f}{fe_m:9.1f}"
              f"{cov:7.1f}{succ:7.1f}{en:10.1f}{fr:8.1f}")
    return base_starved


def sweep_table(tag, key, xs, xlabel, methods=METHODS, main_tag=None, main_x=None):
    print(f"\n--- sweep {xlabel} (tag={tag}) : final acc% ---")
    print(f"{'method':16s}" + "".join(f"{str(x):>8s}" for x in xs))
    for m in methods:
        row = []
        for x in xs:
            if main_tag and x == main_x:
                runs = load(tag=main_tag, method=m, seed=1)
            else:
                runs = load(tag=tag, method=m, **{key: x})
            row.append(np.mean([r["hist"]["acc"][-1] for r in runs]) * 100 if runs else float("nan"))
        print(f"{m:16s}" + "".join(f"{v:8.1f}" for v in row))


def density_table(tag, main_tag):
    print(f"\n--- sweep traffic density (tag={tag}) : final acc% ---")
    dmap = [("low_n60_s1", "low"), ("med_n60_s1", "med"), ("high_n60_s1", "high")]
    print(f"{'method':16s}" + "".join(f"{lab:>8s}" for _, lab in dmap))
    for m in METHODS:
        row = []
        for tr, _ in dmap:
            runs = load(tag=main_tag, method=m, trace=tr, seed=1) if "med" in tr \
                else load(tag=tag, method=m, trace=tr)
            row.append(np.mean([r["hist"]["acc"][-1] for r in runs]) * 100 if runs else float("nan"))
        print(f"{m:16s}" + "".join(f"{v:8.1f}" for v in row))


if __name__ == "__main__":
    for dkey, mt, pf, title in [
        ("raymob8", "rmmain", "rm",
         "Raymobtime beam selection (camera+LiDAR+GPS, 8 Tx sectors, "
         "concat fusion + 20% starvation; chance=12.5%)"),
    ]:
        main_table(mt, title)
        sweep_table(f"{pf}sweepR", "R", [100, 150, 200, 300],
                    "V2V range (m)", main_tag=mt, main_x=200)
        sweep_table(f"{pf}sweepC", "C", [2, 4, 6, 8],
                    "cache capacity", methods=CACHE_METHODS, main_tag=mt, main_x=4)
        density_table(f"{pf}sweepD", mt)
    print("\n(starved% = mean final acc of vehicles starved on one modality;"
          " fed% = the rest. RECD's premise needs starved% > baselines.)")
