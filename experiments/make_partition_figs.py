"""Figures for the partitioned carry-forward scenario (Manhattan grid + zones +
carriers). Proves visually that, in a network split into zones bridged only by
travelling carriers, store-carry-forward delivers a needed encoder to far-zone
vehicles that no-sharing leaves stranded.

  fig_map        - grid roads + 3 zones + vehicle snapshot + one carrier's path
  fig_contact    - V2V contact graph: 3 intra-zone clusters bridged by carriers
  fig_carryfwd   - far-zone needy reception vs round: Local (0%, stranded) vs
                   carry-forward methods (rise to ~100%)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "results", "figs")
os.makedirs(FIGS, exist_ok=True)
TRACE = os.path.join(BASE, "results", "traces", "trace_manh_z3_n60.npz")
plt.rcParams.update({"font.size": 11, "font.family": "serif", "savefig.bbox": "tight"})
ZCOL = ["#1f77b4", "#2ca02c", "#9467bd"]   # zone colors
CARCOL = "#d62728"

d = np.load(TRACE)
pos, zone = d["pos"], d["zone"]
N, T, _ = pos.shape
G, sp = 10, 160.0                  # grid params used to generate the trace


def fig_map():
    import gen_partitioned as gp
    g = gp.build_grid(G, sp)
    fig, ax = plt.subplots(figsize=(5, 5))
    # zone bands (by node column j -> y in [j*sp]); 3 horizontal bands
    span = (G - 1) * sp
    for z in range(3):
        ax.axhspan(z * span / 3 - sp / 2, (z + 1) * span / 3 - sp / 2,
                   color=ZCOL[z], alpha=0.07)
    # grid roads
    for (u, v) in g["edges"]:
        pu, pv = g["node_xy"][u], g["node_xy"][v]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="#cccccc", lw=0.6, zorder=1)
    # vehicle snapshot at mid-sim
    t = T // 2
    for i in range(N):
        if zone[i] < 0:
            continue
        ax.scatter(*pos[i, t], color=ZCOL[zone[i]], s=18, zorder=3)
    # one carrier's full trajectory + position
    car = int(np.where(zone < 0)[0][0])
    ax.plot(pos[car, :, 0], pos[car, :, 1], color=CARCOL, lw=1.4, alpha=0.8,
            zorder=4, label="carrier path")
    ax.scatter(*pos[car, t], color=CARCOL, s=80, marker="*", zorder=5,
               edgecolor="black", label="carrier")
    ax.set_title("Manhattan grid: 3 zones + carrier")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="upper right"); ax.set_aspect("equal")
    fig.savefig(os.path.join(FIGS, "fig_map.pdf")); plt.close(fig)
    print("wrote fig_map.pdf")


def fig_contact(R=100.0):
    # contact if ever within R; node position = mean position
    ever = np.zeros((N, N), bool)
    for t in range(0, T, 3):
        p = pos[:, t]
        dist = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        ever |= dist <= R
    mp = np.nanmean(pos, axis=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(N):
        for j in range(i + 1, N):
            if ever[i, j]:
                cross = (zone[i] != zone[j]) or zone[i] < 0 or zone[j] < 0
                ax.plot([mp[i, 0], mp[j, 0]], [mp[i, 1], mp[j, 1]],
                        color=(CARCOL if cross else "#dddddd"),
                        lw=(1.0 if cross else 0.4), alpha=(0.7 if cross else 0.5),
                        zorder=(3 if cross else 1))
    for i in range(N):
        c = CARCOL if zone[i] < 0 else ZCOL[zone[i]]
        ax.scatter(*mp[i], color=c, s=(70 if zone[i] < 0 else 30),
                   marker=("*" if zone[i] < 0 else "o"),
                   edgecolor="black", lw=0.4, zorder=4)
    ax.set_title("V2V contact graph (red = inter-zone, via carriers)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.savefig(os.path.join(FIGS, "fig_contact.pdf")); plt.close(fig)
    print("wrote fig_contact.pdf")


def fig_carryfwd():
    from engine import run
    needy = [int(i) for i in np.where(zone >= 1)[0]]
    starve = [int(i) for i in range(N) if zone[i] != 0]
    base = dict(trace=TRACE, seed=1, rounds=60, r_v2v=100.0, cache_encoders=4,
                max_out=2, V=50.0, use_gat=False, dataset="fmnist",
                fusion_mode="concat", phi_agg=0.15, gated_agg=False, local_steps=3,
                p_mod=1.0, starve_ids=starve, starve_mod=0, round_sec=10)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    col = {"Local": "#7f7f7f", "DFL-Gossip": "#2ca02c", "LRU-Random": "#ff7f0e",
           "RECD": "#d62728"}
    K = base["rounds"]
    for meth in ["Local", "DFL-Gossip", "LRU-Random", "RECD"]:
        h = run(dict(base, method=meth))
        fr = h["first_recv"]
        recv_round = [fr[f"{i}_0"] for i in needy if f"{i}_0" in fr]
        cov = [100 * sum(r <= k for r in recv_round) / len(needy) for k in range(1, K + 1)]
        ax.plot(range(1, K + 1), cov, label=meth, color=col[meth], lw=1.8)
    ax.set_xlabel("Global round")
    ax.set_ylabel("Far-zone needy reached (%)")
    ax.set_title("Carry-forward delivery to far zones")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGS, "fig_carryfwd.pdf")); plt.close(fig)
    print("wrote fig_carryfwd.pdf")


if __name__ == "__main__":
    fig_map()
    fig_contact()
    fig_carryfwd()
    print("partition figures in", FIGS)
