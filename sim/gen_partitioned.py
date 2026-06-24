"""Synthetic Manhattan-grid mobility with SPATIAL ZONES + a few CARRIERS.

Builds an engine-compatible trace .npz (same schema as the SUMO traces) on a
GxG Manhattan grid -- the "Manhattan mobility model" the paper's table claims.
Most vehicles are confined to one of K horizontal zones (their encoders stay
zone-local); a few CARRIERS roam the whole grid and can physically carry a
cached encoder from one zone to another. This is the regime where store-carry-
forward and mobility-aware caching can actually matter: a needy vehicle in a
far zone receives a useful encoder ONLY if a carrier ferries it there.

Output: results/traces/trace_<tag>.npz with pos/speed/edge_idx/lane_pos,
lengths/headings/centers, feas, trans_keys/vals, edge_count/speed/flow,
stat_window, edge_ids, plus 'zone' (per-vehicle zone, -1 = carrier).
"""
import argparse
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_grid(G, d):
    """GxG intersections, spacing d. Returns directed edges and helpers."""
    node_xy = {}
    for i in range(G):
        for j in range(G):
            node_xy[i * G + j] = np.array([i * d, j * d], float)
    edges = []  # (tail_node, head_node)
    for i in range(G):
        for j in range(G):
            n = i * G + j
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < G and 0 <= nj < G:
                    edges.append((n, ni * G + nj))
    eidx = {e: k for k, e in enumerate(edges)}
    E = len(edges)
    centers = np.zeros((E, 2), np.float32)
    lengths = np.full(E, float(d), np.float32)
    headings = np.zeros(E, np.float32)
    for k, (u, v) in enumerate(edges):
        pu, pv = node_xy[u], node_xy[v]
        centers[k] = (pu + pv) / 2
        headings[k] = np.degrees(np.arctan2(pv[1] - pu[1], pv[0] - pu[0])) % 360
    # successor edges of edge k (tail=head(k)), excluding immediate U-turn
    succ = [[] for _ in range(E)]
    feas = []
    for k, (u, v) in enumerate(edges):
        for (a, b), kk in eidx.items():
            if a == v and b != u:
                succ[k].append(kk); feas.append((k, kk))
    return dict(node_xy=node_xy, edges=edges, eidx=eidx, E=E, centers=centers,
                lengths=lengths, headings=headings, succ=succ,
                feas=np.array(feas, np.int32), G=G, d=d)


def zone_of_node(n, G, K):
    return min(int((n % G) / (G / K)), K - 1)   # by column index j -> horizontal bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=10)
    ap.add_argument("--d", type=float, default=160.0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--zones", type=int, default=3)
    ap.add_argument("--carriers", type=int, default=6)
    ap.add_argument("--dur", type=int, default=600)
    ap.add_argument("--speed", type=float, default=11.0)
    ap.add_argument("--stat-window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="manh_z3_n60")
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    g = build_grid(args.G, args.d)
    E, edges, succ = g["E"], g["edges"], g["succ"]
    K, T, N = args.zones, args.dur, args.n

    # edges fully inside a zone (both endpoints' column in same band)
    ezone = np.array([zone_of_node(edges[k][0], args.G, K) for k in range(E)])
    ezone_head = np.array([zone_of_node(edges[k][1], args.G, K) for k in range(E)])
    in_zone_edges = {z: [k for k in range(E)
                         if ezone[k] == z and ezone_head[k] == z] for z in range(K)}

    # assign vehicles: carriers (zone=-1) roam all; locals confined to a zone
    veh_zone = np.full(N, -1, int)
    carriers = rng.choice(N, size=min(args.carriers, N), replace=False)
    locals_ = [i for i in range(N) if i not in carriers]
    for k, i in enumerate(locals_):
        veh_zone[i] = k % K                       # balanced across zones

    pos = np.zeros((N, T, 2), np.float32)
    speed = np.full((N, T), args.speed, np.float32)
    edge_idx = np.zeros((N, T), np.int32)
    lane_pos = np.zeros((N, T), np.float32)

    W = args.stat_window
    nwin = T // W
    edge_count = np.zeros((nwin, E), np.float32)
    edge_speed = np.zeros((nwin, E), np.float32)
    edge_flow = np.zeros((nwin, E), np.float32)
    trans = {}

    def allowed(i):
        return list(range(E)) if veh_zone[i] < 0 else in_zone_edges[veh_zone[i]]

    # init each vehicle on a random allowed edge
    cur = np.array([rng.choice(allowed(i)) for i in range(N)])
    s = rng.uniform(0, g["lengths"][cur])         # distance along edge
    for t in range(T):
        w = min(t // W, nwin - 1)
        for i in range(N):
            e = cur[i]; s[i] += args.speed
            if s[i] >= g["lengths"][e]:            # reached intersection -> turn
                s[i] -= g["lengths"][e]
                nxt = [k for k in succ[e] if (veh_zone[i] < 0 or
                       (ezone[k] == veh_zone[i] and ezone_head[k] == veh_zone[i]))]
                if not nxt:                        # dead-end inside zone -> U-turn
                    u, v = edges[e]
                    nxt = [g["eidx"][(v, u)]] if (v, u) in g["eidx"] else succ[e]
                ne = int(rng.choice(nxt))
                trans[(e, ne)] = trans.get((e, ne), 0) + 1
                edge_flow[w, ne] += 1
                cur[i] = e = ne
            u, v = edges[e]
            pu, pv = g["node_xy"][u], g["node_xy"][v]
            frac = s[i] / g["lengths"][e]
            pos[i, t] = pu + frac * (pv - pu)
            edge_idx[i, t] = e; lane_pos[i, t] = s[i]
            edge_count[w, e] += 1; edge_speed[w, e] += args.speed

    tk = np.array(list(trans.keys()), np.int32) if trans else np.zeros((0, 2), np.int32)
    tv = np.array(list(trans.values()), np.int32) if trans else np.zeros(0, np.int32)
    out = os.path.join(BASE, "results", "traces", f"trace_{args.tag}.npz")
    np.savez(out, pos=pos, speed=speed, edge_idx=edge_idx, lane_pos=lane_pos,
             lengths=g["lengths"], headings=g["headings"], centers=g["centers"],
             feas=g["feas"], trans_keys=tk, trans_vals=tv,
             edge_count=edge_count, edge_speed=edge_speed, edge_flow=edge_flow,
             stat_window=np.array([W]), edge_ids=np.array([str(e) for e in range(E)]),
             zone=veh_zone)
    print(f"saved {out}: N={N} ({len(carriers)} carriers), E={E}, T={T}, "
          f"zones={K}, grid {args.G}x{args.G}")
    # quick partition check: fraction of vehicle-pairs ever within 150m, intra vs inter zone
    print("zones per vehicle:", np.bincount(veh_zone[veh_zone >= 0], minlength=K),
          "| carriers:", list(carriers))


if __name__ == "__main__":
    main()
