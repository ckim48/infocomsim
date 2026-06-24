"""(a)-(d) time sequence on the REAL San Francisco STREET MAP: a modality encoder
trained at one source vehicle is cached and physically carried across the city
via V2V store-carry-forward. Backdrop = actual SF road polylines from sf.net.xml
(aligned with the SUMO trace coords); red = vehicles currently holding the encoder.
"""
import os
import re
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MPath

# simple side-view car silhouette (body + cabin + 2 wheels), centered at origin
def _car_marker():
    body = [(-1.0, -.32), (1.0, -.32), (1.0, .06), (.55, .06), (.32, .46),
            (-.40, .46), (-.60, .06), (-1.0, .06), (-1.0, -.32)]
    bc = [MPath.MOVETO] + [MPath.LINETO] * 7 + [MPath.CLOSEPOLY]
    verts, codes = list(body), list(bc)
    for cx in (-.58, .58):                      # two wheels
        r = .22
        circ = [(cx + r * np.cos(a), -.32 + r * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, 12)]
        verts += circ + [circ[0]]
        codes += [MPath.MOVETO] + [MPath.LINETO] * 11 + [MPath.CLOSEPOLY]
    return MPath(verts, codes)

CAR = _car_marker()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "results", "figs")
NET = os.path.join(BASE, "scenario", "sf.net.xml")
TRACE = os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz")  # SUMO SF, net-aligned
CACHE = os.path.join(BASE, "results", "sf_journey_med_track.json")
ROUND_SEC = 20
plt.rcParams.update({"font.size": 10, "font.family": "serif", "savefig.bbox": "tight"})


def load_roads():
    """All non-internal lane polylines from sf.net.xml -> list of (M,2) arrays."""
    segs = []
    txt = open(NET).read()
    for m in re.finditer(r'<lane id="(:?[^"]*)"[^>]*shape="([^"]+)"', txt):
        if m.group(1).startswith(":"):           # skip internal junction lanes
            continue
        pts = [p.split(",") for p in m.group(2).split()]
        xy = np.array([[float(a), float(b)] for a, b in pts])
        if len(xy) >= 2:
            for k in range(len(xy) - 1):
                segs.append([xy[k], xy[k + 1]])
    return segs


d = np.load(TRACE)
pos = d["pos"]
N, T, _ = pos.shape
p0 = pos[:, 0]
src = int(np.argmin(p0[:, 0] + p0[:, 1]))        # corner source

if os.path.exists(CACHE):
    track = json.load(open(CACHE)); print("loaded cached track")
else:
    from engine import run
    others = [i for i in range(N) if i != src]    # src = unique good mod0 encoder
    cfg = dict(method="RECD", trace=TRACE, seed=1, rounds=60, r_v2v=150.0,
               cache_encoders=6, max_out=3, V=50.0, use_gat=False, dataset="raymob8",
               fusion_mode="concat", phi_agg=0.15, gated_agg=False, local_steps=3,
               p_mod=1.0, starve_ids=others, starve_mod=0, round_sec=ROUND_SEC,
               track_key=(src, 0))
    track = run(cfg)["track"]
    json.dump(track, open(CACHE, "w"))

roads = load_roads()
print(f"roads: {len(roads)} segments; source veh {src}; "
      f"holders {min(len(s['holders']) for s in track)}..{max(len(s['holders']) for s in track)}")
nh = [len(s["holders"]) for s in track]
mx = max(nh)
panels, used = [], set()
for tg in [1, max(2, int(0.25 * mx)), max(3, int(0.6 * mx)), mx]:
    k = next((i for i in range(len(nh)) if nh[i] >= tg and i not in used), len(nh) - 1)
    used.add(k); panels.append(k)

xs = np.concatenate([np.array(r)[:, 0] for r in roads])
ys = np.concatenate([np.array(r)[:, 1] for r in roads])
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
for ax, k, lab in zip(axes, panels, "abcd"):
    s = track[k]
    ax.add_collection(LineCollection(roads, colors="#cfcfcf", linewidths=0.5, zorder=1))
    t = min(s["round"] * ROUND_SEC, T - 1)
    holders = set(s["holders"])
    others = [i for i in range(N) if i not in holders and i != src]
    ax.scatter(pos[others, t, 0], pos[others, t, 1], marker=CAR, s=130,
               color="#74add1", edgecolor="#2c3e50", lw=0.3, zorder=2)        # cars w/o it
    hp = np.array(s["pos"])
    if len(hp):
        ax.scatter(hp[:, 0], hp[:, 1], marker=CAR, s=170, color="#d62728",
                   edgecolor="black", lw=0.4, zorder=4, label="holds encoder") # cars w/ it
    ax.scatter(*pos[src, t], marker=CAR, s=300, color="#f1a340",
               edgecolor="black", lw=0.8, zorder=6, label="source")           # source car
    ax.set_title(f"({lab}) round {s['round']} — {len(s['holders'])} vehicles hold it")
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Encoder carried across the San Francisco street network via V2V "
             "store-carry-forward (RECD)", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_sf_journey.pdf")); plt.close(fig)
print("wrote fig_sf_journey.pdf, panels at rounds", [track[k]["round"] for k in panels])
