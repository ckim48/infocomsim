"""Scenario figure: SF downtown road network + vehicles + V2V topology.

(a) Full SUMO road network (San Francisco downtown OSM extract) with all
    vehicles at one snapshot and their V2V links (pairs within R^V2V).
(b) Zoom-in showing V2V communication ranges and the resulting contact graph.
Vehicles are colored by their instantaneous V2V degree to show the uneven
contact opportunities motivating road-aware dissemination.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(BASE, "scenario", "sf.net.xml")
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")
FIG = os.path.join(BASE, "figures")
R_V2V = 200.0
T_SNAP = 300

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.linewidth": 1.0,
})


def load_lanes():
    """All non-internal lane polylines from the SUMO network."""
    segs = []
    for edge in ET.parse(NET).getroot().findall("edge"):
        if edge.get("function") == "internal" or edge.get("id", "").startswith(":"):
            continue
        for lane in edge.findall("lane"):
            pts = [tuple(map(float, p.split(",")))
                   for p in lane.get("shape", "").split()]
            if len(pts) >= 2:
                segs.append(np.array(pts))
    return segs


def main():
    lanes = load_lanes()
    d = np.load(TRACE, allow_pickle=True)
    P = d["pos"][:, T_SNAP, :]                      # [N,2]
    N = len(P)
    dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    links = [(i, j) for i in range(N) for j in range(i + 1, N)
             if dist[i, j] <= R_V2V]
    deg = np.zeros(N)
    for i, j in links:
        deg[i] += 1
        deg[j] += 1

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6))

    for ax in axes:
        ax.add_collection(LineCollection(lanes, colors="0.78", linewidths=0.4))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # (a) full network + all vehicles + V2V links
    ax = axes[0]
    lc = [[P[i], P[j]] for i, j in links]
    ax.add_collection(LineCollection(lc, colors="#4C72B0", linewidths=0.5,
                                     alpha=0.5))
    sc = ax.scatter(P[:, 0], P[:, 1], c=deg, cmap="YlOrRd", s=22,
                    edgecolors="k", linewidths=0.3, zorder=3, vmin=0)
    ax.set_title("(a) SF downtown: 60 vehicles + V2V links", y=-0.10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("V2V degree", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # (b) zoom on the densest cluster
    ax = axes[1]
    c = P[deg.argmax()]
    W = 600
    sel = [k for k in range(N)
           if abs(P[k, 0] - c[0]) <= W and abs(P[k, 1] - c[1]) <= W]
    for k in sel:
        ax.add_patch(Circle(P[k], R_V2V, fill=False, ec="#4C72B0",
                            lw=0.3, alpha=0.25))
    lc2 = [[P[i], P[j]] for i, j in links if i in sel and j in sel]
    ax.add_collection(LineCollection(lc2, colors="#4C72B0", linewidths=0.7,
                                     alpha=0.7))
    ax.scatter(P[sel, 0], P[sel, 1], c=deg[sel], cmap="YlOrRd", s=40,
               edgecolors="k", linewidths=0.4, zorder=3, vmin=0)
    ax.set_xlim(c[0] - W, c[0] + W)
    ax.set_ylim(c[1] - W, c[1] + W)
    ax.set_title(r"(b) zoom: $R^{\mathrm{V2V}}=200$ m ranges & contacts",
                 y=-0.10)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_scenario.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"lanes={len(lanes)} links={len(links)} "
          f"deg[min/mean/max]={deg.min():.0f}/{deg.mean():.1f}/{deg.max():.0f}")
    print("saved figures/fig_scenario.pdf / .png")


if __name__ == "__main__":
    main()
