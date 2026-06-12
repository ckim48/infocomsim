"""Generate mobility traces from SUMO for the DFL simulation.

Injects N participant vehicles that drive continuously (re-routed to a new
random destination when close to arrival) on top of background traffic, and
records per-second positions/segments/speeds plus per-segment traffic
statistics and segment-to-segment transition counts.

Output: results/traces/trace_<density>_n<N>_s<seed>.npz
"""
import os
import sys
import argparse
import random
import numpy as np

SUMO_HOME = os.environ.get("SUMO_HOME", "/opt/homebrew/opt/sumo/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402
import sumolib  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(BASE, "scenario", "sf.net.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", choices=["low", "med", "high"], default="med")
    ap.add_argument("--n", type=int, default=60, help="participant vehicles")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--duration", type=int, default=1200)
    ap.add_argument("--stat-window", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    net = sumolib.net.readNet(NET)
    edges = sorted(
        [e for e in net.getEdges() if e.allows("passenger") and e.getLength() > 20],
        key=lambda e: e.getID(),
    )
    eid2idx = {e.getID(): i for i, e in enumerate(edges)}
    E = len(edges)

    trips = os.path.join(BASE, "scenario", f"trips_{args.density}.trips.xml")
    sumo_cmd = [
        os.path.join(SUMO_HOME, "..", "..", "bin", "sumo")
        if os.path.exists(os.path.join(SUMO_HOME, "..", "..", "bin", "sumo"))
        else "sumo",
        "-n", NET, "-r", trips,
        "--seed", str(args.seed),
        "--step-length", "1",
        "--no-warnings", "--no-step-log",
        "--time-to-teleport", "120",
        "--ignore-route-errors",
        "--max-depart-delay", "60",
    ]
    traci.start(sumo_cmd)

    participants = [f"fl_{i}" for i in range(args.n)]
    long_edges = [e for e in edges if e.getLength() > 50]

    def random_far_route(from_edge_id=None):
        """Pick a random origin (if needed) and a reachable far destination."""
        for _ in range(50):
            src = from_edge_id or rng.choice(long_edges).getID()
            dst = rng.choice(long_edges).getID()
            if dst == src:
                continue
            route = traci.simulation.findRoute(src, dst, "DEFAULT_VEHTYPE", -1.0, 0)
            if route.edges and len(route.edges) >= 5:
                return src, route.edges
        return None, None

    # inject participants during the first warmup seconds
    inserted = 0
    depart_times = {}
    last_add = {}        # vid -> time of last traci.vehicle.add
    departed_after_add = {}  # vid -> True once seen in the running set
    for vid in participants:
        depart_times[vid] = rng.uniform(0, args.warmup * 0.8)

    T = args.duration
    N = args.n
    pos = np.full((N, T, 2), np.nan, dtype=np.float32)
    speed = np.zeros((N, T), dtype=np.float32)
    edge_idx = np.full((N, T), -1, dtype=np.int32)
    lane_pos = np.zeros((N, T), dtype=np.float32)
    heading_sign = np.ones((N, T), dtype=np.int8)

    W = args.stat_window
    nwin = T // W
    edge_count = np.zeros((nwin, E), dtype=np.float32)   # vehicle-seconds
    edge_speed = np.zeros((nwin, E), dtype=np.float32)   # speed sum
    edge_flow = np.zeros((nwin, E), dtype=np.float32)    # entries
    trans_count = np.zeros((0,), dtype=np.int32)         # placeholder
    transitions = {}                                     # (from_idx,to_idx) -> count

    last_edge_all = {}   # vehID -> last real edge idx (all vehicles, for stats)
    last_edge_p = {}     # participant -> last real edge idx

    pid = {v: i for i, v in enumerate(participants)}

    t_sim = 0
    rec_start = args.warmup
    while t_sim < rec_start + T:
        # insert participants
        for vid in participants:
            if vid in depart_times and t_sim >= depart_times[vid]:
                src, route_edges = random_far_route()
                if route_edges:
                    rid = f"r_{vid}_{t_sim}"
                    traci.route.add(rid, route_edges)
                    traci.vehicle.add(vid, rid, departLane="best", departSpeed="max")
                    traci.vehicle.setColor(vid, (255, 0, 0, 255))
                    del depart_times[vid]
                    last_add[vid] = t_sim
                    departed_after_add[vid] = False
                    inserted += 1

        traci.simulationStep()
        t_sim += 1
        cur = set(traci.vehicle.getIDList())

        # keep participants alive: extend route when near the end
        for vid in participants:
            if vid in cur:
                departed_after_add[vid] = True
                ridx = traci.vehicle.getRouteIndex(vid)
                redges = traci.vehicle.getRoute(vid)
                if len(redges) - ridx <= 2:
                    cur_edge = redges[-1]
                    _, new_edges = random_far_route(cur_edge)
                    if new_edges:
                        traci.vehicle.setRoute(vid, [redges[ridx]] + list(new_edges))
            elif vid not in depart_times and vid in last_add:
                if departed_after_add[vid]:
                    # arrived or teleported out after driving: re-insert soon
                    depart_times[vid] = t_sim + 5
                elif t_sim - last_add[vid] > 90:
                    # pending insertion was discarded (max-depart-delay): retry
                    depart_times[vid] = t_sim + 5

        if t_sim <= rec_start:
            continue
        t = t_sim - rec_start - 1
        w = min(t // W, nwin - 1)

        for vid in cur:
            road = traci.vehicle.getRoadID(vid)
            ei = eid2idx.get(road, -1)
            sp = traci.vehicle.getSpeed(vid)
            if ei >= 0:
                edge_count[w, ei] += 1.0
                edge_speed[w, ei] += sp
                le = last_edge_all.get(vid, -1)
                if le != ei:
                    edge_flow[w, ei] += 1.0
                    if le >= 0:
                        transitions[(le, ei)] = transitions.get((le, ei), 0) + 1
                last_edge_all[vid] = ei
            if vid in pid:
                i = pid[vid]
                x, y = traci.vehicle.getPosition(vid)
                pos[i, t] = (x, y)
                speed[i, t] = sp
                if ei >= 0:
                    edge_idx[i, t] = ei
                    lane_pos[i, t] = traci.vehicle.getLanePosition(vid)
                    last_edge_p[vid] = ei
                else:
                    edge_idx[i, t] = last_edge_p.get(vid, -1)

    traci.close()

    tk = np.array(list(transitions.keys()), dtype=np.int32).reshape(-1, 2)
    tv = np.array(list(transitions.values()), dtype=np.int32)

    # edge metadata
    lengths = np.array([e.getLength() for e in edges], dtype=np.float32)
    headings = np.zeros(E, dtype=np.float32)
    centers = np.zeros((E, 2), dtype=np.float32)
    for i, e in enumerate(edges):
        shp = e.getShape()
        (x0, y0), (x1, y1) = shp[0], shp[-1]
        headings[i] = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360.0
        centers[i] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    # feasible transitions from net topology
    feas = []
    for i, e in enumerate(edges):
        for nxt in e.getOutgoing():
            j = eid2idx.get(nxt.getID(), -1)
            if j >= 0:
                feas.append((i, j))
    feas = np.array(feas, dtype=np.int32)

    out_dir = os.path.join(BASE, "results", "traces")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"trace_{args.density}_n{args.n}_s{args.seed}.npz")
    np.savez_compressed(
        out,
        pos=pos, speed=speed, edge_idx=edge_idx, lane_pos=lane_pos,
        edge_count=edge_count, edge_speed=edge_speed, edge_flow=edge_flow,
        trans_keys=tk, trans_vals=tv,
        lengths=lengths, headings=headings, centers=centers, feas=feas,
        edge_ids=np.array([e.getID() for e in edges]),
        stat_window=np.array([W]),
    )
    valid = np.isfinite(pos[:, :, 0]).mean()
    print(f"saved {out}; participant presence={valid:.3f}; "
          f"transitions={len(tv)}; edges={E}")


if __name__ == "__main__":
    main()
