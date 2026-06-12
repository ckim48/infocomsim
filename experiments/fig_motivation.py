"""Motivation figure (Fig. 1 of the paper).

(a) nuScenes data volume per sensing modality -- published dataset statistics
    from Caesar et al., "nuScenes: A multimodal dataset for autonomous
    driving," CVPR 2020 (1.4M camera images, 390k LiDAR sweeps, 1.4M radar
    sweeps, 1.4M annotated objects over 1000 scenes; 11.6% night scenes).
(b) PLACEHOLDER panel -- per-drive captured-object statistics require the
    nuScenes metadata; run your own analysis and replace. Rendered here with
    a clearly marked "requires nuScenes" annotation so the figure compiles.
(c) V2V contacts are concentrated on a small portion of road segments
    (Lorenz-style curve, real SF network + SUMO).
(d) Per-vehicle V2V contact opportunities are uneven (histogram, SUMO).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "sim"))
from road_graph import RoadGraph  # noqa: E402
import comm  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def contact_stats():
    rg = RoadGraph(os.path.join(BASE, "results", "traces",
                                "trace_med_n60_s1.npz"))
    prm = comm.V2VParams()
    prm.R = 200.0
    seg_contacts = np.zeros(rg.E)
    veh_contacts = np.zeros(rg.N)
    active = {}  # (i, j) -> start time
    for t in range(0, rg.T, 2):
        d = comm.distances(rg, t)
        within = d <= prm.R
        for i in range(rg.N):
            for j in range(i + 1, rg.N):
                if within[i, j] and (i, j) not in active:
                    active[(i, j)] = t
                    e = rg.edge_idx[i, t]
                    if e >= 0:
                        seg_contacts[e] += 1
                    veh_contacts[i] += 1
                    veh_contacts[j] += 1
                elif not within[i, j] and (i, j) in active:
                    del active[(i, j)]
    return seg_contacts, veh_contacts


def main():
    seg, veh = contact_stats()

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75))

    # (a) nuScenes modality volumes (published statistics)
    ax = axes[0]
    mods = ["Camera", "LiDAR", "Radar"]
    counts = [1.4e6, 3.9e5, 1.3e6]
    ax.bar(mods, counts, color=["#4C72B0", "#DD8452", "#55A868"], width=0.6)
    ax.set_ylabel("Samples in nuScenes")
    ax.set_title("(a) Data per modality", y=-0.42)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # (b) placeholder -- requires nuScenes metadata
    ax = axes[1]
    ax.text(0.5, 0.5, "requires nuScenes\nmetadata analysis\n(see README)",
            ha="center", va="center", fontsize=7, style="italic",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) Objects per drive", y=-0.42)

    # (c) Lorenz curve of contacts over segments
    ax = axes[2]
    s = np.sort(seg)[::-1]
    cum = np.cumsum(s) / max(s.sum(), 1)
    x = np.arange(1, len(s) + 1) / len(s) * 100
    ax.plot(x, cum * 100, "-", color="#4C72B0", lw=1.5)
    ax.axvline(20, color="gray", ls=":", lw=0.8)
    f20 = cum[int(0.2 * len(s)) - 1] * 100
    ax.annotate(f"top 20% segs:\n{f20:.0f}% of contacts", xy=(20, f20),
                xytext=(38, 52), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("Road segments (%)")
    ax.set_ylabel("V2V contacts (%)")
    ax.set_title("(c) Contact concentration", y=-0.55)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)

    # (d) per-vehicle contact histogram
    ax = axes[3]
    ax.hist(veh, bins=14, color="#DD8452", edgecolor="white", lw=0.4)
    ax.set_xlabel("V2V contacts per vehicle")
    ax.set_ylabel("Vehicles")
    ax.set_title("(d) Uneven opportunities", y=-0.55)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    out = os.path.join(BASE, "Figures", "fig_motivation.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)
    print(f"top-20% segments host {np.cumsum(np.sort(seg)[::-1])[int(0.2*len(seg))-1]/max(seg.sum(),1)*100:.1f}% of contacts")
    print(f"vehicle contacts: min={veh.min():.0f} median={np.median(veh):.0f} max={veh.max():.0f}")


if __name__ == "__main__":
    main()
