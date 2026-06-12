"""IEEE-style result figures from results/runs/*.json."""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "results", "runs")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})

METHODS = ["RECD", "DFL-Gossip", "LRU-Random", "Mobility-Greedy"]
STYLE = {
    "RECD":            dict(color="#C44E52", marker="o", label="RECD (proposed)"),
    "DFL-Gossip":      dict(color="#4C72B0", marker="s", label="DFL-Gossip"),
    "LRU-Random":      dict(color="#55A868", marker="^", label="LRU-Random"),
    "Mobility-Greedy": dict(color="#8172B3", marker="d", label="Mobility-Greedy"),
}
SINGLE = (3.5, 2.3)


def load(tag=None, method=None, trace=None, R=None, C=None, V=None, seed=None):
    out = []
    for f in glob.glob(os.path.join(RUNS, "*.json")):
        b = os.path.basename(f)[:-5]
        parts = b.split("_")
        # <tag>_<method>_<trace...>_R<r>_C<c>_V<v>_s<seed>
        d = dict(tag=parts[0], method=parts[1])
        d["R"] = int([p for p in parts if p.startswith("R")][-1][1:])
        d["C"] = int([p for p in parts if p.startswith("C")][-1][1:])
        d["V"] = int([p for p in parts if p.startswith("V") and p[1:].isdigit()][-1][1:])
        d["seed"] = int(parts[-1][1:])
        d["trace"] = "_".join(parts[2:-4])
        ok = ((tag is None or d["tag"] == tag)
              and (method is None or d["method"] == method)
              and (trace is None or d["trace"] == trace)
              and (R is None or d["R"] == R)
              and (C is None or d["C"] == C)
              and (V is None or d["V"] == V)
              and (seed is None or d["seed"] == seed))
        if ok:
            with open(f) as fh:
                j = json.load(fh)
            j["meta"] = d
            out.append(j)
    return out


def finish(ax, xlab, ylab, legend=True, loc="best"):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(loc=loc, frameon=False)


def fig_acc_main():
    fig, ax = plt.subplots(figsize=SINGLE)
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        accs = np.array([r["hist"]["acc"] for r in runs])
        rounds = runs[0]["hist"]["acc_round"]
        mu, sd = accs.mean(0) * 100, accs.std(0) * 100
        ax.plot(rounds, mu, **STYLE[m])
        ax.fill_between(rounds, mu - sd, mu + sd, color=STYLE[m]["color"],
                        alpha=0.15, lw=0)
    finish(ax, "Training round", "Test accuracy (%)", loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_acc_main.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_network():
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0))
    # coverage vs round
    ax = axes[0, 0]
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        cov = np.array([r["hist"]["coverage"] for r in runs]).mean(0) * 100
        ax.plot(np.arange(1, len(cov) + 1), cov, **STYLE[m])
    finish(ax, "Training round", "Encoder coverage (%)", loc="lower right")
    ax.set_title("(a)", y=-0.45)

    # success ratio
    ax = axes[0, 1]
    xs = np.arange(len(METHODS))
    for k, m in enumerate(METHODS):
        runs = load(tag="main", method=m)
        if not runs:
            continue
        sr = [np.sum(r["hist"]["succ"]) / max(np.sum(r["hist"]["att"]), 1)
              for r in runs]
        ax.bar(k, np.mean(sr) * 100, yerr=np.std(sr) * 100, width=0.55,
               color=STYLE[m]["color"], error_kw=dict(lw=0.8))
    ax.set_xticks(xs)
    ax.set_xticklabels(["RECD", "Gossip", "LRU-Rnd", "Mob-Grd"])
    finish(ax, "", "Tx success ratio (%)", legend=False)
    ax.set_title("(b)", y=-0.45)

    # cumulative energy
    ax = axes[1, 0]
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        en = np.array([np.cumsum(r["hist"]["energy"]) for r in runs]).mean(0)
        ax.plot(np.arange(1, len(en) + 1), en, **STYLE[m])
    finish(ax, "Training round", "Cumulative energy (J)", loc="upper left")
    ax.set_title("(c)", y=-0.45)

    # accuracy per unit energy (efficiency)
    ax = axes[1, 1]
    for k, m in enumerate(METHODS):
        runs = load(tag="main", method=m)
        if not runs:
            continue
        eff = [r["hist"]["acc"][-1] * 100 / (np.sum(r["hist"]["energy"]) + 1e-9)
               for r in runs]
        ax.bar(k, np.mean(eff), yerr=np.std(eff), width=0.55,
               color=STYLE[m]["color"], error_kw=dict(lw=0.8))
    ax.set_xticks(xs)
    ax.set_xticklabels(["RECD", "Gossip", "LRU-Rnd", "Mob-Grd"])
    finish(ax, "", "Accuracy per energy\n(% / J)", legend=False)
    ax.set_title("(d)", y=-0.45)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "fig_network.pdf"), bbox_inches="tight")
    plt.close(fig)


def _sweep_plot(tag, key, xvals, xlabel, fname, methods=METHODS,
                fixed_from_main=None):
    """Final accuracy vs swept parameter; pulls the main-config point too."""
    fig, ax = plt.subplots(figsize=SINGLE)
    for m in methods:
        pts = []
        for x in xvals:
            if fixed_from_main is not None and x == fixed_from_main:
                runs = load(tag="main", method=m, seed=1)
            else:
                runs = load(tag=tag, method=m, **{key: x})
            if runs:
                pts.append((x, np.mean([r["hist"]["acc"][-1] for r in runs]) * 100))
        if pts:
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], **STYLE[m])
    finish(ax, xlabel, "Final test accuracy (%)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def fig_acc_har():
    """Accuracy vs rounds on UCI-HAR (real multimodal dataset)."""
    fig, ax = plt.subplots(figsize=SINGLE)
    any_run = False
    for m in METHODS:
        runs = load(tag="harmain", method=m)
        if not runs:
            continue
        any_run = True
        accs = np.array([r["hist"]["acc"] for r in runs])
        rounds = runs[0]["hist"]["acc_round"]
        mu, sd = accs.mean(0) * 100, accs.std(0) * 100
        ax.plot(rounds, mu, **STYLE[m])
        ax.fill_between(rounds, mu - sd, mu + sd, color=STYLE[m]["color"],
                        alpha=0.15, lw=0)
    if not any_run:
        plt.close(fig)
        print("  (no harmain runs yet; skipping fig_acc_har)")
        return
    finish(ax, "Training round", "Test accuracy (%)", loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_acc_har.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_pareto():
    """Headline: final accuracy vs total transmission energy (3 seeds)."""
    fig, ax = plt.subplots(figsize=SINGLE)
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        accs = [r["hist"]["acc"][-1] * 100 for r in runs]
        ens = [np.sum(r["hist"]["energy"]) / 1e3 for r in runs]
        ax.errorbar(np.mean(ens), np.mean(accs),
                    xerr=np.std(ens), yerr=np.std(accs),
                    color=STYLE[m]["color"], marker=STYLE[m]["marker"],
                    ms=7, capsize=2, lw=1.2, label=STYLE[m]["label"])
    ax.annotate("better", xy=(0.07, 0.93), xycoords="axes fraction",
                fontsize=7, style="italic")
    ax.annotate("", xy=(0.03, 0.97), xytext=(0.16, 0.84),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.9))
    finish(ax, "Total transmission energy (kJ)", "Final test accuracy (%)",
           loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sweepR2():
    """Two-panel R sweep: accuracy and energy vs V2V range."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1))
    Rs = [100, 150, 200, 300]
    for m in METHODS:
        accs, ens = [], []
        xs = []
        for R in Rs:
            runs = (load(tag="main", method=m, seed=1, R=R) if R == 200
                    else load(tag="sweepR", method=m, R=R))
            if runs:
                xs.append(R)
                accs.append(np.mean([r["hist"]["acc"][-1] for r in runs]) * 100)
                ens.append(np.mean([np.sum(r["hist"]["energy"]) for r in runs]) / 1e3)
        if xs:
            axes[0].plot(xs, accs, **STYLE[m])
            axes[1].plot(xs, ens, **STYLE[m])
    finish(axes[0], "V2V range $R^{\\mathrm{V2V}}$ (m)",
           "Final test accuracy (%)", loc="lower right")
    finish(axes[1], "V2V range $R^{\\mathrm{V2V}}$ (m)",
           "Total energy (kJ)", legend=False)
    axes[0].set_title("(a)", y=-0.5)
    axes[1].set_title("(b)", y=-0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "fig_sweepR.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_cdf():
    """Per-vehicle final accuracy CDF + breakdown by sensing quality."""
    runs_by_m = {m: load(tag="main", method=m) for m in METHODS}
    if not any("acc_per_veh" in r["hist"] for rs in runs_by_m.values()
               for r in rs):
        print("  (per-vehicle accuracy not yet recorded; skipping fig_cdf)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1))
    ax = axes[0]
    for m in METHODS:
        accs = np.concatenate(
            [r["hist"]["acc_per_veh"] for r in runs_by_m[m]
             if "acc_per_veh" in r["hist"]]) * 100
        if len(accs):
            xs = np.sort(accs)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), **STYLE[m])
    finish(ax, "Per-vehicle final accuracy (%)", "CDF", loc="upper left")
    ax.set_title("(a)", y=-0.5)

    ax = axes[1]
    w = 0.2
    groups = ["Low-quality\nsensing", "Scarce data\n($D{<}200$)", "All"]
    for k, m in enumerate(METHODS):
        lo_q, lo_d, allv = [], [], []
        for r in runs_by_m[m]:
            h = r["hist"]
            if "acc_per_veh" not in h:
                continue
            for a, meta in zip(h["acc_per_veh"], h["veh_meta"]):
                allv.append(a)
                if any(q < 0.7 for q in meta["Q"].values()):
                    lo_q.append(a)
                if min(meta["D"].values()) < 200:
                    lo_d.append(a)
        ys = [np.mean(g) * 100 if g else 0 for g in (lo_q, lo_d, allv)]
        ax.bar(np.arange(3) + (k - 1.5) * w, ys, width=w,
               color=STYLE[m]["color"], label=STYLE[m]["label"])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(groups, fontsize=6.5)
    finish(ax, "", "Final accuracy (%)", legend=False)
    ax.set_title("(b)", y=-0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "fig_cdf.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sweepN():
    fig, ax = plt.subplots(figsize=SINGLE)
    nmap = [("med_n30_s1", 30), ("med_n60_s1", 60), ("med_n100_s1", 100)]
    for m in METHODS:
        pts = []
        for tr, n in nmap:
            runs = load(tag="main", method=m, trace=tr, seed=1) if n == 60 \
                else load(tag="sweepN", method=m, trace=tr)
            if runs:
                pts.append((n, np.mean([r["hist"]["acc"][-1] for r in runs]) * 100))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], **STYLE[m])
    finish(ax, "Number of vehicles", "Final test accuracy (%)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_sweepN.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sweepD():
    fig, ax = plt.subplots(figsize=SINGLE)
    dmap = [("low_n60_s1", "Low"), ("med_n60_s1", "Med"), ("high_n60_s1", "High")]
    w = 0.2
    for k, m in enumerate(METHODS):
        ys = []
        for tr, lab in dmap:
            runs = load(tag="main", method=m, trace=tr, seed=1) if "med" in tr \
                else load(tag="sweepD", method=m, trace=tr)
            ys.append(np.mean([r["hist"]["acc"][-1] for r in runs]) * 100
                      if runs else 0)
        ax.bar(np.arange(3) + (k - 1.5) * w, ys, width=w,
               color=STYLE[m]["color"], label=STYLE[m]["label"])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([d[1] for d in dmap])
    finish(ax, "Background traffic density", "Final test accuracy (%)",
           loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_sweepD.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sweepV():
    fig, ax = plt.subplots(figsize=SINGLE)
    Vs = [5, 20, 50, 100, 200]
    dlv, yq = [], []
    for V in Vs:
        runs = load(tag="main", method="RECD", seed=1) if V == 50 \
            else load(tag="sweepV", V=V)
        if runs:
            dlv.append(np.mean([np.mean(r["hist"]["succ"]) for r in runs]))
            yq.append(np.mean([np.mean(r["hist"]["y_queue"]) for r in runs]))
    ax.plot(Vs[:len(dlv)], dlv, color="#C44E52", marker="o",
            label="Encoder deliveries/round")
    ax.set_xscale("log")
    ax2 = ax.twinx()
    ax2.plot(Vs[:len(yq)], yq, color="#4C72B0", marker="s", ls="--",
             label="Mean queue backlog")
    ax2.set_ylabel("Mean virtual queue $Y$", color="#4C72B0")
    ax.set_ylabel("Encoder deliveries per round", color="#C44E52")
    ax.set_xlabel("Lyapunov parameter $V$")
    ax.grid(alpha=0.3, lw=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_sweepV.pdf"), bbox_inches="tight")
    plt.close(fig)


def table_summary():
    rows = []
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        acc = np.mean([r["hist"]["acc"][-1] for r in runs]) * 100
        sr = np.mean([np.sum(r["hist"]["succ"]) / max(np.sum(r["hist"]["att"]), 1)
                      for r in runs]) * 100
        en = np.mean([np.sum(r["hist"]["energy"]) for r in runs])
        fr = np.mean([r["hist"]["mean_first_recv"] for r in runs])
        jn = np.mean([r["hist"]["jain"] for r in runs])
        st = np.mean([np.mean(r["hist"]["staleness"]) for r in runs])
        rows.append((m, acc, sr, en, fr, jn, st))
    print(f"{'method':18s} {'acc%':>6s} {'succ%':>6s} {'energy J':>9s} "
          f"{'1st-recv':>8s} {'Jain':>5s} {'stale':>6s}")
    for r in rows:
        print(f"{r[0]:18s} {r[1]:6.2f} {r[2]:6.1f} {r[3]:9.1f} "
              f"{r[4]:8.2f} {r[5]:5.3f} {r[6]:6.2f}")
    return rows


if __name__ == "__main__":
    fig_acc_main()
    fig_acc_har()
    fig_network()
    fig_pareto()
    fig_sweepR2()
    fig_cdf()
    _sweep_plot("sweepC", "C", [2, 4, 6, 8],
                "Cache capacity (encoders)", "fig_sweepC.pdf",
                methods=["RECD", "LRU-Random", "Mobility-Greedy"],
                fixed_from_main=4)
    fig_sweepN()
    fig_sweepD()
    fig_sweepV()
    table_summary()
    print("figures written to", FIG)
