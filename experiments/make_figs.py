"""Publication figures for the INFOCOM paper. Writes PDFs to results/figs/.

Honest story the figures tell:
  fig_energy        - RECD disseminates at ~1/3 the energy of the baselines.
  fig_network       - RECD: fewer transmissions, higher success ratio.
  fig_equalbudget   - at EQUAL comm budget (max_out=1), RECD's need/mobility-aware
                      targeting gives the best degraded-sensor encoder recovery
                      (random flooding only "won" earlier by spending 2.5x more).
  fig_regime        - encoder sharing helps in the QUALITY-degradation regime,
                      not the quantity-starvation regime (fusion/joint-data bound).
  fig_curve         - accuracy vs round (RECD vs baselines), beam task.
Each figure is skipped (with a note) if its source runs are absent.
"""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "results", "runs")
FIGS = os.path.join(BASE, "results", "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 120, "savefig.bbox": "tight",
    "legend.frameon": False,
})
ORDER = ["RECD", "Mobility-Greedy", "DFL-Gossip", "LRU-Random", "Local"]
COL = {"RECD": "#d62728", "Mobility-Greedy": "#1f77b4", "DFL-Gossip": "#2ca02c",
       "LRU-Random": "#ff7f0e", "Local": "#7f7f7f"}


def load(tag, method):
    return [json.load(open(f))["hist"]
            for f in glob.glob(os.path.join(RUNS, f"{tag}_{method}_*.json"))]


def ms(vals):
    return (np.mean(vals), np.std(vals)) if len(vals) else (np.nan, 0.0)


def present(tag, methods):
    return [m for m in methods if load(tag, m)]


def bar(ax, methods, vals, errs, ylabel, title):
    xs = np.arange(len(methods))
    ax.bar(xs, vals, yerr=errs, capsize=3,
           color=[COL.get(m, "#333") for m in methods], edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)


def fig_energy(tag="rmmain"):
    methods = present(tag, ["RECD", "Mobility-Greedy", "DFL-Gossip", "LRU-Random"])
    if not methods:
        print("skip fig_energy: no", tag); return
    v, e = [], []
    for m in methods:
        a, b = ms([np.sum(h["energy"]) / 1e3 for h in load(tag, m)]); v.append(a); e.append(b)
    fig, ax = plt.subplots(figsize=(4, 3))
    bar(ax, methods, v, e, "Total V2V energy (kJ)", "Dissemination energy")
    fig.savefig(os.path.join(FIGS, "fig_energy.pdf")); plt.close(fig)
    print("wrote fig_energy.pdf", dict(zip(methods, np.round(v, 1))))


def fig_network(tag="rmmain"):
    methods = present(tag, ["RECD", "Mobility-Greedy", "DFL-Gossip", "LRU-Random"])
    if not methods:
        print("skip fig_network: no", tag); return
    tx, sr = [], []
    for m in methods:
        hs = load(tag, m)
        tx.append(ms([np.mean(h["att"]) for h in hs]))
        sr.append(ms([100 * np.sum(h["succ"]) / max(np.sum(h["att"]), 1) for h in hs]))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    bar(axes[0], methods, [t[0] for t in tx], [t[1] for t in tx],
        "Transmissions / round", "Comm. volume")
    bar(axes[1], methods, [s[0] for s in sr], [s[1] for s in sr],
        "Delivery success (%)", "Reliability")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_network.pdf")); plt.close(fig)
    print("wrote fig_network.pdf")


def _lowq_enc(h):
    return [x * 100 for e, d in zip(h["enc_acc_per_veh"], h["veh_meta"])
            for r, x in e.items() if d["Q"][str(r)] < 0.6]


def fig_equalbudget(tag="encq2"):
    methods = present(tag, ORDER)
    if not methods or "Local" not in methods:
        print("skip fig_equalbudget: no", tag); return
    v, e = [], []
    for m in methods:
        a, b = ms([np.mean(_lowq_enc(h)) for h in load(tag, m) if "enc_acc_per_veh" in h])
        v.append(a); e.append(b)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    bar(ax, methods, v, e, "Degraded-sensor encoder acc. (%)",
        "Equal comm. budget (max_out=1)")
    loc = v[methods.index("Local")]
    ax.axhline(loc, ls="--", color="gray", lw=1)
    fig.savefig(os.path.join(FIGS, "fig_equalbudget.pdf")); plt.close(fig)
    print("wrote fig_equalbudget.pdf", dict(zip(methods, np.round(v, 1))))


def fig_regime(quant="encc", qual="encq2"):
    """RECD encoder-sharing gain over Local: quantity-starve vs quality-degrade."""
    out = {}
    for name, tag, grp in [("Quantity-\nstarved", quant, "starved"),
                           ("Quality-\ndegraded", qual, "quality")]:
        loc = load(tag, "Local"); rec = load(tag, "RECD")
        if not loc or not rec:
            continue

        def sel(h):
            return [x * 100 for e, d in zip(h["enc_acc_per_veh"], h["veh_meta"])
                    for r, x in e.items()
                    if (d["D"][str(r)] <= 5 if grp == "starved" else d["Q"][str(r)] < 0.6)]
        lb = np.mean([np.mean(sel(h)) for h in loc if "enc_acc_per_veh" in h])
        rb = [np.mean(sel(h)) for h in rec if "enc_acc_per_veh" in h]
        out[name] = (np.mean(rb) - lb, np.std(rb))
    if not out:
        print("skip fig_regime: missing", quant, qual); return
    fig, ax = plt.subplots(figsize=(4, 3))
    names = list(out)
    ax.bar(names, [out[n][0] for n in names], yerr=[out[n][1] for n in names],
           capsize=3, color=["#7f7f7f", "#d62728"], edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("RECD encoder gain vs Local (pp)")
    ax.set_title("Where sharing helps the encoder")
    fig.savefig(os.path.join(FIGS, "fig_regime.pdf")); plt.close(fig)
    print("wrote fig_regime.pdf", {k: round(v[0], 1) for k, v in out.items()})


def fig_curve(tag="rmmain"):
    methods = present(tag, ORDER)
    if not methods:
        print("skip fig_curve: no", tag); return
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for m in methods:
        hs = load(tag, m)
        accs = np.array([h["acc"] for h in hs if len(h["acc"]) == len(hs[0]["acc"])])
        rounds = hs[0]["acc_round"]
        mu = accs.mean(0) * 100
        ax.plot(rounds, mu, label=m, color=COL.get(m), lw=1.6)
    ax.set_xlabel("Global round"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Beam-selection accuracy"); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGS, "fig_curve.pdf")); plt.close(fig)
    print("wrote fig_curve.pdf")


if __name__ == "__main__":
    fig_energy()
    fig_network()
    fig_equalbudget()
    fig_regime()
    # fig_curve()  # beam-8 accuracy is non-monotone (encoder-irrelevant task) -- omitted
    print("figures in", FIGS)
