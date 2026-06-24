"""Next-segment mobility prediction accuracy (pivoted paper's core claim).

HierGAT (vehicle-specific, with trajectory history) vs GAT (shared) vs Markov.
Metrics: top-1 over all transitions and over BRANCHING segments only (>=3
feasible successors, where prediction is non-trivial). Time split: train early,
test late.
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import gat_predictor as G
from road_graph import RoadGraph, markov_kernel, calibrate_psi

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = (sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(BASE, "results", "traces", "trace_med_n60_s1.npz"))
R = 200.0
EPOCHS = 200

rg = RoadGraph(TRACE)
src, dst, dirlab = G._feas_tensors(rg)
row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(rg.feas)}
rows_by_src = {}
for i, (a, b) in enumerate(rg.feas):
    rows_by_src.setdefault(int(a), []).append(i)
branch = {e for e, rws in rows_by_src.items() if len(rws) >= 3}

by_t = G._collect_transitions(rg)
times = sorted(by_t)
cut = times[int(0.6 * len(times))]
train_t = [t for t in times if t < cut]
test_t = [t for t in times if t >= cut]
rng = np.random.RandomState(0)
def samp(ts, n):
    return ts if len(ts) <= n else sorted(rng.choice(ts, n, replace=False).tolist())
train_s = samp(train_t, 160)
test_s = samp(test_t, 160)
print(f"transitions: train {sum(len(by_t[t]) for t in train_s)} ({len(train_s)} t) | "
      f"test {sum(len(by_t[t]) for t in test_s)} ({len(test_s)} t); "
      f"branching segs={len(branch)}", flush=True)


def train(vehicle_specific, seed=0):
    torch.manual_seed(seed)
    m = G.HierGAT(vehicle_specific=vehicle_specific)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        opt.zero_grad(); loss = torch.zeros(()); ntr = 0
        for t in train_s:
            hr, hv = m.embed(rg, t, R, src, dst, dirlab)
            o = m.trans_logits(hr, hv, src, dst, dirlab)
            for j, e, en in by_t[t]:
                row = row_of.get((e, en))
                if row is None:
                    continue
                pi = G.scatter_softmax(o[j], src, rg.E)
                loss = loss - torch.log(pi[row] + 1e-12); ntr += 1
        if ntr == 0:
            break
        (loss / ntr).backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def eval_gat(m):
    c1 = tot = bc1 = btot = 0
    for t in test_s:
        hr, hv = m.embed(rg, t, R, src, dst, dirlab)
        o = m.trans_logits(hr, hv, src, dst, dirlab)
        for j, e, en in by_t[t]:
            idxs = rows_by_src.get(e, [])
            tr = row_of.get((e, en))
            if len(idxs) < 2 or tr is None:
                continue
            top = idxs[int(torch.argmax(o[j, idxs]))]
            tot += 1; c1 += int(top == tr)
            if e in branch:
                btot += 1; bc1 += int(top == tr)
    return 100*c1/tot, 100*bc1/max(btot, 1), tot, btot


def eval_markov():
    psi = calibrate_psi(rg)
    c1 = tot = bc1 = btot = 0
    for t in test_s:
        P = markov_kernel(rg, t, psi)
        for j, e, en in by_t[t]:
            outs = rg.out_edges[e]
            if len(outs) < 2:
                continue
            top = outs[int(np.argmax(P[e, outs]))]
            tot += 1; c1 += int(top == en)
            if e in branch:
                btot += 1; bc1 += int(top == en)
    return 100*c1/tot, 100*bc1/max(btot, 1), tot, btot


print("\ntraining HierGAT (vehicle-specific + history) ...", flush=True)
t0 = time.time(); mh = train(True); print(f"  {time.time()-t0:.0f}s", flush=True)
print("training GAT (shared) ...", flush=True)
t0 = time.time(); ms = train(False); print(f"  {time.time()-t0:.0f}s", flush=True)

h1, hb, tot, btot = eval_gat(mh)
s1, sb, _, _ = eval_gat(ms)
m1, mb, _, _ = eval_markov()
print("\n===== next-segment prediction top-1 accuracy (held-out future) =====")
print(f'{"predictor":30}{"all":>8}{"branching":>11}')
print(f'{"Markov (shared, analytic)":30}{m1:8.1f}{mb:11.1f}')
print(f'{"GAT (shared)":30}{s1:8.1f}{sb:11.1f}')
print(f'{"HierGAT (veh-specific+hist)":30}{h1:8.1f}{hb:11.1f}')
print(f'(n_all={tot}, n_branching={btot})')
import json
json.dump(dict(markov=[m1, mb], shared_gat=[s1, sb], hiergat=[h1, hb],
               n_all=tot, n_branch=btot),
          open(os.path.join(BASE, "results", "eval_prediction.json"), "w"), indent=1)
print("saved results/eval_prediction.json")
