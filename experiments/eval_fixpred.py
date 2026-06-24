"""FAIR real-data mobility prediction: predict the cab's segment ~60s ahead
(next GPS fix), with NO shortest-path interpolation. Tests whether conditioning
on heading / trajectory history / vehicle identity beats a pure current-segment
empirical prior on REAL San Francisco taxi movement.

  Empirical prior  : argmax P(seg_next | seg_cur) from training fixes
  Context model    : MLP( seg, prev_seg, heading, cab-id ) scoring candidates
Metric: top-1 over all transitions and over branching segments (>=3 observed
successors). Time split: train earlier fixes, test later.
"""
import os, sys, glob, time, collections, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import netgraph as NG

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(BASE, "scenario", "sf.net.xml")
DATA = os.path.join(BASE, "results", "data", "realgps", "cabspottingdata")
LON0, LAT0, LON1, LAT1 = -122.413743, 37.778247, -122.389654, 37.807433
T0, DUR = 1212760000, 14400        # ~4 h window (more data for a fair test)
MAXM = 50.0                         # map-match accept distance (m)

rg = NG.build_road_graph(NET)
E = len(rg["edge_ids"]); cen = rg["centers"]

def match(xs, ys):
    """vectorized-ish nearest segment via center pre-filter + fine polyline dist."""
    out = []
    for x, y in zip(xs, ys):
        dc = np.hypot(cen[:, 0] - x, cen[:, 1] - y)
        cand = np.argsort(dc)[:12]
        ei, best, _ = NG.match_point(x, y, rg, cand=cand)
        out.append((ei, best))
    return out

print("map-matching cab fixes in window ...", flush=True)
t0 = time.time()
cabs = []          # list of (cab_id, [(t, seg, x, y)])
files = sorted(glob.glob(os.path.join(DATA, "new_*.txt")))
for ci, fp in enumerate(files):
    rows = []
    for line in open(fp):
        f = line.split()
        if len(f) < 4:
            continue
        lat, lon, t = float(f[0]), float(f[1]), float(f[3])
        if T0 <= t <= T0 + DUR and LON0 <= lon <= LON1 and LAT0 <= lat <= LAT1:
            rows.append((t, lat, lon))
    if len(rows) < 5:
        continue
    rows.sort()
    xs, ys = NG.latlon_to_net(np.array([r[1] for r in rows]),
                              np.array([r[2] for r in rows]), NET)
    seq = []
    for k, (t, _, _) in enumerate(rows):
        ei, d = match([xs[k]], [ys[k]])[0]
        if d <= MAXM:
            seq.append((t, ei, float(xs[k]), float(ys[k])))
    if len(seq) >= 4:
        cabs.append(seq)
print(f"  {len(cabs)} cabs, {sum(len(s) for s in cabs)} matched fixes, "
      f"{time.time()-t0:.0f}s", flush=True)

# build transition samples: (t, cab_idx, seg, prev_seg, heading, seg_next)
samples = []
for ci, seq in enumerate(cabs):
    for k in range(1, len(seq) - 1):
        seg, segn = seq[k][1], seq[k + 1][1]
        if seg == segn:
            continue
        dx, dy = seq[k][2] - seq[k-1][2], seq[k][3] - seq[k-1][3]
        head = np.arctan2(dy, dx)
        samples.append((seq[k][0], ci, seg, seq[k-1][1], head, segn))
samples.sort()
cut = samples[int(0.6 * len(samples))][0]
train = [s for s in samples if s[0] < cut]
test = [s for s in samples if s[0] >= cut]
print(f"transitions: train {len(train)} | test {len(test)}", flush=True)

# candidate successors per segment (from training) + empirical prior counts
succ = collections.defaultdict(collections.Counter)
for _, ci, seg, ps, h, segn in train:
    succ[seg][segn] += 1
branch = {s for s, c in succ.items() if len(c) >= 3}

def eval_prior(rows):
    c1 = tot = bc = bt = 0
    for _, ci, seg, ps, h, segn in rows:
        cand = succ.get(seg)
        if not cand or len(cand) < 2 or segn not in cand:
            continue                      # same closed candidate set as the model
        pred = max(cand, key=lambda k: cand[k])
        tot += 1; c1 += int(pred == segn)
        if seg in branch:
            bt += 1; bc += int(pred == segn)
    return 100*c1/max(tot,1), 100*bc/max(bt,1), tot, bt

# ---- context model
NC = len(cabs)
class CtxModel(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.seg = nn.Embedding(E, d)
        self.cab = nn.Embedding(NC, d)
        self.ctx = nn.Sequential(nn.Linear(3*d+2, 64), nn.ReLU(), nn.Linear(64, d))
        self.score = nn.Sequential(nn.Linear(2*d, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, seg, ps, head, cab, cand, prior_lp):  # cand: [B,C] padded, -1 invalid
        h = self.ctx(torch.cat([self.seg(seg), self.seg(ps), self.cab(cab),
                                torch.stack([torch.sin(head), torch.cos(head)], 1)], 1))
        ce = self.seg(cand.clamp(min=0))                       # [B,C,d]
        he = h.unsqueeze(1).expand(-1, cand.shape[1], -1)
        s = self.score(torch.cat([he, ce], 2)).squeeze(-1)     # [B,C]
        s = s + prior_lp                                       # base-rate prior
        s = s.masked_fill(cand < 0, -1e9)
        return s

def make_batch(rows):
    seg, ps, head, cab, cands, tgt = [], [], [], [], [], []
    valid = []
    for r in rows:
        _, ci, s, p, h, sn = r
        cand = list(succ.get(s, {}).keys())
        if len(cand) < 2 or sn not in cand:
            continue
        valid.append((s, p, h, ci, cand, cand.index(sn)))
    maxc = max(len(v[4]) for v in valid)
    for s, p, h, ci, cand, ti in valid:
        pad = cand + [-1]*(maxc-len(cand))
        seg.append(s); ps.append(p); head.append(h); cab.append(ci)
        cands.append(pad); tgt.append(ti)
    plp = []
    for s_, p_, h_, ci_, cand_, ti_ in valid:
        cnt = succ.get(s_, {})
        tot = sum(cnt.values()) + 1e-9
        row = [np.log(cnt.get(c, 0) / tot + 1e-9) for c in cand_]
        row += [-1e9] * (maxc - len(row))
        plp.append(row)
    return (torch.tensor(seg), torch.tensor(ps), torch.tensor(head, dtype=torch.float32),
            torch.tensor(cab), torch.tensor(cands), torch.tensor(tgt),
            torch.tensor(plp, dtype=torch.float32))

torch.manual_seed(0)
m = CtxModel(); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
tb = make_batch(train)
for ep in range(400):
    opt.zero_grad()
    s = m(tb[0], tb[1], tb[2], tb[3], tb[4], tb[6])
    loss = nn.functional.cross_entropy(s, tb[5])
    loss.backward(); opt.step()

@torch.no_grad()
def eval_model(rows):
    vb = make_batch(rows)
    s = m(vb[0], vb[1], vb[2], vb[3], vb[4], vb[6])
    pred = s.argmax(1)
    segs = vb[0].numpy()
    corr = (pred == vb[5]).numpy()
    tot = len(corr); c1 = corr.sum()
    bmask = np.array([sg in branch for sg in segs])
    return 100*c1/max(tot,1), 100*corr[bmask].sum()/max(bmask.sum(),1), tot, int(bmask.sum())

p1, pb, ptot, pbt = eval_prior(test)
m1, mb, mtot, mbt = eval_model(test)
print("\n===== FAIR fix-level prediction (real SF taxi, no interpolation) =====")
print(f'{"predictor":34}{"top-1":>8}{"branching":>11}')
print(f'{"Empirical prior P(next|cur)":34}{p1:8.1f}{pb:11.1f}   (n={ptot},{pbt})')
print(f'{"Context model (head+hist+cab)":34}{m1:8.1f}{mb:11.1f}   (n={mtot},{mbt})')
import json
json.dump(dict(prior=[p1,pb], context=[m1,mb], n=[ptot,pbt]),
          open(os.path.join(BASE, "results", "eval_fixpred.json"), "w"), indent=1)
print("saved results/eval_fixpred.json")
