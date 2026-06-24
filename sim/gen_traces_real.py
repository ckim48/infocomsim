"""Turn real taxi GPS (cabspotting format) into a RoadGraph .npz trace.

Cabspotting: a directory of files 'new_<id>.txt', each line
"lat lon occupancy unixtime" (newest first). We map-match each cab's fixes to
SF road segments (netgraph), reconstruct the segment path between sparse fixes
via shortest road-graph paths, resample to a 1 s grid over a chosen window for
N cabs, and recompute per-window traffic stats and transition counts.

Usage:
  python sim/gen_traces_real.py --data results/data/cabspotting \
      --out results/traces/trace_realSF_n60.npz --n 60 --t0 <unix> --dur 1200
"""
import os, sys, glob, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netgraph as NG

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(BASE, "scenario", "sf.net.xml")


def shortest_path(rg, a, b, maxhops=8):
    """BFS over feasible transitions; returns [a,...,b] or None."""
    if a == b:
        return [a]
    adj = rg.setdefault("_adj", None)
    if adj is None:
        adj = collections.defaultdict(list)
        for s, d in rg["feas"]:
            adj[int(s)].append(int(d))
        rg["_adj"] = adj
    q = collections.deque([[a]]); seen = {a}
    while q:
        path = q.popleft()
        if len(path) > maxhops:
            continue
        for nxt in adj[path[-1]]:
            if nxt == b:
                return path + [b]
            if nxt not in seen:
                seen.add(nxt); q.append(path + [nxt])
    return None


def load_cab(fp):
    out = []
    for line in open(fp):
        f = line.split()
        if len(f) >= 4:
            try:
                out.append((float(f[3]), float(f[0]), float(f[1])))  # t,lat,lon
            except ValueError:
                pass
    out.sort()
    return out  # ascending time


def build(args):
    rg = NG.build_road_graph(NET)
    E = len(rg["edge_ids"])
    files = sorted(glob.glob(os.path.join(args.data, "new_*.txt")) or
                   glob.glob(os.path.join(args.data, "*.txt")))
    if not files:
        sys.exit(f"no cab files in {args.data}")
    # choose window
    cabs = []
    for fp in files:
        pts = load_cab(fp)
        if pts:
            cabs.append(pts)
    if args.t0 <= 0:
        # default: a 1 h window starting at the global median start time
        starts = sorted(p[0][0] for p in cabs)
        args.t0 = int(starts[len(starts) // 2])
    t1 = args.t0 + args.dur
    # keep cabs with enough fixes in the window
    chosen = []
    for pts in cabs:
        w = [p for p in pts if args.t0 <= p[0] <= t1]
        if len(w) >= args.min_fix:
            chosen.append(w)
        if len(chosen) >= args.n:
            break
    if len(chosen) < args.n:
        print(f"warning: only {len(chosen)} cabs have >= {args.min_fix} fixes")
    N = len(chosen)
    T = args.dur
    pos = np.zeros((N, T, 2), np.float32)
    edge_idx = np.full((N, T), -1, np.int32)
    lane_pos = np.zeros((N, T), np.float32)
    speed = np.zeros((N, T), np.float32)
    trans = collections.Counter()

    for vi, w in enumerate(chosen):
        ts = np.array([p[0] for p in w])
        xs, ys = NG.latlon_to_net(np.array([p[1] for p in w]),
                                  np.array([p[2] for p in w]), NET)
        # map-match each fix
        seq = []  # (t, ei, x, y, lane_pos)
        last_ei = None
        for k in range(len(w)):
            cand = None
            if last_ei is not None:
                cand = [last_ei] + rg["_adj"].get(last_ei, []) if "_adj" in rg \
                    else None
            ei, d, lp = NG.match_point(xs[k], ys[k], rg)
            if d > args.max_match_m:
                continue
            seq.append((ts[k], ei, xs[k], ys[k], lp))
            last_ei = ei
        if len(seq) < 2:
            continue
        # fill 1s grid: hold segment between fixes; record transitions
        for a in range(len(seq) - 1):
            (ta, ea, xa, ya, la) = seq[a]
            (tb, eb, xb, yb, lb) = seq[a + 1]
            i0 = int(max(0, ta - args.t0)); i1 = int(min(T, tb - args.t0))
            if i1 <= i0:
                continue
            dist = float(np.hypot(xb - xa, yb - ya))
            spd = dist / max(tb - ta, 1.0)
            if ea != eb:
                path = shortest_path(rg, ea, eb) or [ea, eb]
            else:
                path = [ea]
            cen = rg["centers"]; ln = rg["lengths"]; M = len(path)
            for i in range(i0, i1):
                f = (i - i0) / max(i1 - i0, 1)
                e = path[min(int(f * M), M - 1)]
                edge_idx[vi, i] = e
                pos[vi, i] = cen[e]
                lane_pos[vi, i] = 0.5 * ln[e]
                speed[vi, i] = spd
            for s, dd in zip(path[:-1], path[1:]):
                trans[(s, dd)] += 1

    # per-window traffic stats
    W = args.stat_window
    nwin = (T + W - 1) // W
    edge_count = np.zeros((nwin, E), np.float32)
    edge_speed = np.zeros((nwin, E), np.float32)
    edge_flow = np.zeros((nwin, E), np.float32)
    for i in range(T):
        wj = i // W
        for vi in range(N):
            e = edge_idx[vi, i]
            if e >= 0:
                edge_count[wj, e] += 1
                edge_speed[wj, e] += speed[vi, i]
                edge_flow[wj, e] += 1
    tk = np.array(list(trans.keys()), np.int32) if trans else np.zeros((0, 2), np.int32)
    tv = np.array(list(trans.values()), np.int32) if trans else np.zeros((0,), np.int32)

    np.savez(args.out,
             pos=pos, speed=speed, edge_idx=edge_idx, lane_pos=lane_pos,
             lengths=rg["lengths"], headings=rg["headings"], centers=rg["centers"],
             feas=rg["feas"], trans_keys=tk, trans_vals=tv,
             edge_count=edge_count, edge_speed=edge_speed, edge_flow=edge_flow,
             edge_ids=rg["edge_ids"], stat_window=np.array([W], np.int64))
    cov = (edge_idx >= 0).mean()
    print(f"saved {args.out}: N={N} T={T} E={E} feas={len(rg['feas'])} "
          f"trans={len(tk)} grid-coverage={cov:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cabspotting dir (new_*.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--t0", type=int, default=0, help="window start unixtime")
    ap.add_argument("--dur", type=int, default=1200)
    ap.add_argument("--stat-window", type=int, default=10)
    ap.add_argument("--min-fix", type=int, default=10)
    ap.add_argument("--max-match-m", type=float, default=60.0)
    build(ap.parse_args())
