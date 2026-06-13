"""Convergence-per-energy and energy-reduction figures (IEEE style).

RECD reaches the same accuracy as the benchmarks at a fraction of the
transmission energy. Per *round* the accuracy curves overlap, so the honest
way to show the advantage is accuracy vs. cumulative transmission energy
(left figure) and the cumulative-energy gap itself (right figure).
"""
import os
import numpy as np

from plots import load, STYLE, METHODS, FIG, SINGLE
import matplotlib.pyplot as plt


def finish(ax, xlab, ylab, legend=True, loc="best"):
    """Local copy of plots.finish, compatible with matplotlib < 3.4."""
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(loc=loc, frameon=False)


def _seed_mean(runs, key):
    """Stack a per-round hist array across seeds, return mean and std."""
    arr = np.array([r["hist"][key] for r in runs], dtype=float)
    return arr.mean(0), arr.std(0)


def _cum_energy_kj(runs):
    """Cumulative transmission energy (kJ) per round, averaged over seeds."""
    e = np.array([np.cumsum(r["hist"]["energy"]) for r in runs], dtype=float)
    return e.mean(0) / 1000.0, e.std(0) / 1000.0


def fig_acc_vs_energy():
    """Accuracy vs cumulative transmission energy: RECD reaches the target
    accuracy far to the left (much less energy spent)."""
    fig, ax = plt.subplots(figsize=SINGLE)
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        acc_mu, acc_sd = _seed_mean(runs, "acc")
        acc_mu, acc_sd = acc_mu * 100, acc_sd * 100
        rounds = runs[0]["hist"]["acc_round"]
        cum_mu, _ = _cum_energy_kj(runs)
        # cumulative energy at each accuracy-sample round (round r -> index r-1)
        x = np.array([cum_mu[r - 1] for r in rounds])
        ax.plot(x, acc_mu, **STYLE[m])
        ax.fill_between(x, acc_mu - acc_sd, acc_mu + acc_sd,
                        color=STYLE[m]["color"], alpha=0.15, lw=0)
    finish(ax, "Cumulative transmission energy (kJ)", "Test accuracy (%)",
           loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_acc_vs_energy.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_energy():
    """(a) cumulative energy vs round, (b) total energy with % reduction."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))

    # (a) cumulative energy growth
    ax = axes[0]
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        mu, sd = _cum_energy_kj(runs)
        rounds = np.arange(1, len(mu) + 1)
        ax.plot(rounds, mu, **STYLE[m])
        ax.fill_between(rounds, mu - sd, mu + sd,
                        color=STYLE[m]["color"], alpha=0.15, lw=0)
    finish(ax, "Training round", "Cumulative energy (kJ)", loc="upper left")
    ax.set_title("(a)", y=-0.42)

    # (b) total energy bars + reduction vs the cheapest baseline
    ax = axes[1]
    totals = {}
    for m in METHODS:
        runs = load(tag="main", method=m)
        if not runs:
            continue
        t = [np.sum(r["hist"]["energy"]) / 1000.0 for r in runs]
        totals[m] = (np.mean(t), np.std(t))
    baseline = min((v[0] for k, v in totals.items() if k != "RECD"))
    xs = np.arange(len(METHODS))
    for k, m in enumerate(METHODS):
        if m not in totals:
            continue
        mu, sd = totals[m]
        ax.bar(k, mu, yerr=sd, width=0.6, color=STYLE[m]["color"],
               error_kw=dict(lw=0.8))
        if m == "RECD":
            red = (1 - mu / baseline) * 100
            ax.annotate(f"-{red:.0f}%", (k, mu), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8,
                        color=STYLE[m]["color"], fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["RECD", "Gossip", "LRU-Rnd", "Mob-Grd"])
    finish(ax, "", "Total Tx energy (kJ)", legend=False)
    ax.set_title("(b)", y=-0.42)

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_energy.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_acc_vs_energy()
    fig_energy()
    print("saved fig_acc_vs_energy.pdf, fig_energy.pdf")
