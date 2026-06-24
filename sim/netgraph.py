"""Build the road graph directly from a SUMO .net.xml (no SUMO/sumolib needed)
and map-match GPS (lat/lon) onto its segments. Used to turn real taxi GPS
traces into the .npz format consumed by RoadGraph.
"""
import os
import re
import numpy as np
import xml.etree.ElementTree as ET

N_DIR = 8


def _parse_location(net):
    loc = ET.parse(net).getroot().find("location")
    ox, oy = map(float, loc.get("netOffset").split(","))
    return ox, oy, loc.get("projParameter")


def build_road_graph(net, min_len=20.0):
    """Return dict: edge_ids, lengths, headings, centers, shapes (polylines),
    feas (E-edge transitions), eid2idx. Passenger road edges only (len>min)."""
    root = ET.parse(net).getroot()
    eids, shapes, lengths, headings, centers = [], [], [], [], []
    eid2idx = {}
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if edge.get("function") == "internal" or eid.startswith(":"):
            continue
        lanes = edge.findall("lane")
        if not lanes:
            continue
        dis = lanes[0].get("disallow", "")
        allow = lanes[0].get("allow", "")
        if "passenger" in dis:
            continue
        if allow and "passenger" not in allow and "all" not in allow:
            continue
        shp = edge.get("shape") or lanes[0].get("shape")
        if not shp:
            continue
        pts = np.array([[float(v) for v in p.split(",")] for p in shp.split()])
        if len(pts) < 2:
            continue
        seg = np.diff(pts, axis=0)
        L = float(np.sqrt((seg ** 2).sum(1)).sum())
        if L < min_len:
            continue
        d = pts[-1] - pts[0]
        head = float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)
        eid2idx[eid] = len(eids)
        eids.append(eid); shapes.append(pts); lengths.append(L)
        headings.append(head); centers.append(pts.mean(0))
    # feasible transitions from <connection from to>
    feas = []
    seen = set()
    for c in root.findall("connection"):
        a, b = c.get("from"), c.get("to")
        if a in eid2idx and b in eid2idx and (a, b) not in seen and a != b:
            seen.add((a, b)); feas.append((eid2idx[a], eid2idx[b]))
    return dict(
        edge_ids=np.array(eids, dtype="U28"),
        lengths=np.array(lengths, np.float32),
        headings=np.array(headings, np.float32),
        centers=np.array(centers, np.float32),
        shapes=shapes,
        feas=np.array(feas, np.int32),
        eid2idx=eid2idx,
    )


def latlon_to_net(lat, lon, net):
    """Convert WGS84 lat/lon arrays to SUMO net coordinates (meters)."""
    from pyproj import Transformer
    ox, oy, _ = _parse_location(net)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)  # UTM 10N
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x) + ox, np.asarray(y) + oy


def _pt_seg_dist(p, a, b):
    ab = b - a; t = 0.0
    denom = (ab * ab).sum()
    if denom > 1e-9:
        t = np.clip(((p - a) * ab).sum() / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.sqrt(((p - proj) ** 2).sum())), proj, t


def match_point(x, y, rg, cand=None):
    """Nearest road segment to net-point (x,y). Returns (edge_idx, dist, lane_pos)."""
    p = np.array([x, y]); best = (None, 1e18, 0.0)
    idxs = cand if cand is not None else range(len(rg["shapes"]))
    for ei in idxs:
        pts = rg["shapes"][ei]; acc = 0.0
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            d, _, t = _pt_seg_dist(p, a, b)
            seglen = float(np.sqrt(((b - a) ** 2).sum()))
            if d < best[1]:
                best = (ei, d, acc + t * seglen)
            acc += seglen
    return best
