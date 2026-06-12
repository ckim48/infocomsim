"""Print the headline numbers for the Performance Evaluation section."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plots import load, METHODS  # noqa: E402


def final_acc(tag, method, **kw):
    runs = load(tag=tag, method=method, **kw)
    if not runs:
        return None
    return np.mean([r["hist"]["acc"][-1] for r in runs]) * 100


def main():
    print("=== main comparison (3 seeds) ===")
    accs = {}
    for m in METHODS:
        runs = load(tag="main", method=m, trace="med_n60_s1") + \
               load(tag="main", method=m, trace="med_n60_s2") + \
               load(tag="main", method=m, trace="med_n60_s3")
        a = [r["hist"]["acc"][-1] * 100 for r in runs]
        sr = [np.sum(r["hist"]["succ"]) / max(np.sum(r["hist"]["att"]), 1) * 100
              for r in runs]
        en = [np.sum(r["hist"]["energy"]) for r in runs]
        eff = [ai / ei for ai, ei in zip(a, en)]
        accs[m] = np.mean(a)
        print(f"{m:18s} acc={np.mean(a):.2f}±{np.std(a):.2f}%  "
              f"succ={np.mean(sr):.1f}%  energy={np.mean(en):.0f}J  "
              f"eff={np.mean(eff)*1000:.2f} (%/kJ)")
    if "RECD" in accs:
        for m in METHODS[1:]:
            if m in accs:
                print(f"  RECD - {m}: +{accs['RECD']-accs[m]:.2f} pp")

    print("\n=== ablation: GAT vs Markov fallback ===")
    a_gat = final_acc("main", "RECD", trace="med_n60_s1", seed=1)
    a_mk = final_acc("ablation", "RECD")
    if a_gat and a_mk:
        print(f"GAT {a_gat:.2f}%  Markov {a_mk:.2f}%  delta {a_gat-a_mk:+.2f} pp")

    print("\n=== R sweep (seed 1) ===")
    for R in [100, 150, 200, 300]:
        row = []
        for m in METHODS:
            a = (final_acc("main", m, trace="med_n60_s1", seed=1, R=R)
                 if R == 200 else final_acc("sweepR", m, R=R))
            row.append(f"{m}:{a:.1f}" if a else f"{m}:-")
        print(f"R={R}: " + "  ".join(row))

    print("\n=== V sweep (RECD) ===")
    for V in [5, 20, 50, 100, 200]:
        runs = (load(tag="main", method="RECD", trace="med_n60_s1", seed=1)
                if V == 50 else load(tag="sweepV", V=V))
        if runs:
            a = np.mean([r["hist"]["acc"][-1] for r in runs]) * 100
            y = np.mean([np.mean(r["hist"]["y_queue"]) for r in runs])
            z = np.mean([np.mean(r["hist"]["z_queue"]) for r in runs])
            print(f"V={V}: acc={a:.2f}%  Ybar={y:.3f}  Zbar={z:.3f}")


if __name__ == "__main__":
    main()
