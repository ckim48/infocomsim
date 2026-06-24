"""Hierarchical GAT mobility predictor (Sec. "Hierarchical GAT-Based Solution").

Pipeline (per round k):
  road-segment GAT            -> H^road  (Eq. hgat_road_representation)
  road-conditioned veh layer  -> Hbar^veh (bipartite via incidence B)
  V2V vehicle layer           -> H^veh   (Eq. hgat_v2v_representation)
  per-vehicle transition head -> pi_{j,e,e'} (Eq. hgat_transition_probability)
  one-hot propagation         -> Gamma_j (Eq. hgat_road_reachability)

Unlike the old single shared kernel, transition probabilities are
VEHICLE-SPECIFIC: each vehicle j conditions the road-transition head on its own
embedding h_j^veh, which carries its lane state, road context, and V2V context.
Trained offline on per-vehicle (current-seg -> next-seg) transitions from the
SUMO trace.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from road_graph import N_DIR

DG = 32    # graph embedding dim (d_g)
EMB = 8    # direction-label embedding


def scatter_softmax(e, src, n_nodes):
    """Softmax of edge scores e grouped by their src node (per-src normalize)."""
    mx = torch.full((n_nodes,), -1e30, device=e.device)
    mx = mx.scatter_reduce(0, src, e, reduce="amax", include_self=True)
    ex = torch.exp(e - mx[src])
    den = torch.zeros(n_nodes, device=e.device).index_add_(0, src, ex)
    return ex / (den[src] + 1e-12)


class GATLayer(nn.Module):
    """Single-head GAT: attention from src over its dst neighbours, aggregate."""

    def __init__(self, in_dim, out_dim, edge_extra=0):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.att = nn.Linear(2 * out_dim + edge_extra, 1)

    def forward(self, h, src, dst, n_nodes, edge_feat=None):
        hw = self.W(h)
        parts = [hw[src], hw[dst]]
        if edge_feat is not None:
            parts.append(edge_feat)
        e = F.leaky_relu(self.att(torch.cat(parts, dim=1))).squeeze(-1)
        alpha = scatter_softmax(e, src, n_nodes)
        agg = torch.zeros_like(hw)
        agg.index_add_(0, src, alpha.unsqueeze(1) * hw[dst])
        return F.relu(hw + agg)


def veh_features(rg, t, R):
    """Vehicle features x_i^veh(t) = [ell_i, s_i, n_i, T_nbr] (normalized)."""
    seg = rg.edge_idx[:, t]
    lp = rg.lane_pos[:, t]
    seglen = np.where(seg >= 0, rg.lengths[np.clip(seg, 0, rg.E - 1)], 1.0)
    ell = np.clip(lp / np.maximum(seglen, 1e-3), 0.0, 1.0)
    tp = max(t - 1, 0)
    s = np.sign(lp - rg.lane_pos[:, tp])
    s[s == 0] = 1.0
    P = rg.pos[:, t, :]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, 1e9)
    nbr = d <= R
    n = nbr.sum(1)
    sp = rg.speed[:, t]
    rel = np.abs(sp[:, None] - sp[None, :]) + 0.5
    tcon = np.where(nbr, (R - d) / rel, 0.0)
    with np.errstate(invalid="ignore"):
        tnbr = np.where(n > 0, tcon.sum(1) / np.maximum(n, 1), 0.0)
    x = np.stack([ell, (s + 1) / 2.0, n / max(rg.N, 1),
                  np.clip(tnbr / 30.0, 0, 1)], axis=1)
    return torch.tensor(x, dtype=torch.float32)


def com_edges(rg, t, R):
    """Directed V2V edges (both ways) among vehicles within range at time t."""
    P = rg.pos[:, t, :]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, 1e9)
    i, j = np.where(d <= R)
    return (torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long))


def prev_segs(rg):
    """[N,T] previous DISTINCT road segment for each vehicle (-1 if none).

    Lets the predictor condition on the turn just taken (trajectory history),
    which a memoryless Markov kernel cannot use.
    """
    if hasattr(rg, "_prevseg"):
        return rg._prevseg
    E = rg.edge_idx
    prev = np.full(E.shape, -1, dtype=np.int64)
    for j in range(E.shape[0]):
        last, cur = -1, -1
        for t in range(E.shape[1]):
            e = int(E[j, t])
            if e >= 0 and e != cur:
                last, cur = cur, e
            prev[j, t] = last
    rg._prevseg = prev
    return prev


class HierGAT(nn.Module):
    def __init__(self, road_in=6, veh_in=4, vehicle_specific=True):
        super().__init__()
        self.vehicle_specific = vehicle_specific
        self.dir_emb = nn.Embedding(N_DIR, EMB)
        # road-segment GAT
        self.road_in = nn.Linear(road_in, DG)
        self.road_gat = nn.ModuleList([GATLayer(DG, DG, EMB) for _ in range(2)])
        # vehicle input + road-conditioned (bipartite, single segment per veh)
        self.veh_in = nn.Linear(veh_in, DG)
        self.rc = nn.Linear(3 * DG, DG)   # veh + current-seg + prev-seg context
        # V2V vehicle GAT
        self.v2v_gat = GATLayer(DG, DG)
        # per-vehicle road-transition head: w_o^T [h_j | h_e | h_e' | d_delta]
        self.head = nn.Linear(3 * DG + EMB, 1)

    def embed(self, rg, t, R, src, dst, dirlab):
        """Return (h_road [E,DG], h_veh [N,DG]) for round t."""
        xr = torch.tensor(rg.features(t), dtype=torch.float32)
        hr = F.relu(self.road_in(xr))
        de = self.dir_emb(dirlab)
        for layer in self.road_gat:
            hr = layer(hr, src, dst, rg.E, edge_feat=de)
        # road-conditioned vehicle layer (each veh attends to its segment)
        xv = veh_features(rg, t, R)
        hv0 = F.relu(self.veh_in(xv))
        seg = torch.tensor(np.clip(rg.edge_idx[:, t], 0, rg.E - 1),
                           dtype=torch.long)
        pv = prev_segs(rg)[:, t]
        pmask = torch.tensor((pv >= 0).astype(np.float32)).unsqueeze(1)
        pseg = torch.tensor(np.clip(pv, 0, rg.E - 1), dtype=torch.long)
        h_prev = hr[pseg] * pmask
        hv_rc = F.relu(self.rc(torch.cat([hv0, hr[seg], h_prev], dim=1)))
        # V2V vehicle layer
        ci, cj = com_edges(rg, t, R)
        if len(ci) > 0:
            hv = self.v2v_gat(hv_rc, ci, cj, rg.N)
        else:
            hv = hv_rc
        return hr, hv

    def trans_logits(self, hr, hv, src, dst, dirlab):
        """Per-vehicle transition scores o_{j,e,e'} over feasible edges -> [N,F]."""
        de = self.dir_emb(dirlab)
        road_part = torch.cat([hr[src], hr[dst], de], dim=1)   # [F, 2DG+EMB]
        N, Fn = hv.shape[0], road_part.shape[0]
        if self.vehicle_specific:
            hv_e = hv.unsqueeze(1).expand(N, Fn, hv.shape[1])  # [N,F,DG]
        else:  # shared kernel: drop per-vehicle conditioning
            hv_e = torch.zeros(N, Fn, hv.shape[1], device=hv.device)
        rp_e = road_part.unsqueeze(0).expand(N, Fn, road_part.shape[1])
        o = self.head(torch.cat([hv_e, rp_e], dim=2)).squeeze(-1)  # [N,F]
        return o


def _feas_tensors(rg):
    src = torch.tensor(rg.feas[:, 0], dtype=torch.long)
    dst = torch.tensor(rg.feas[:, 1], dtype=torch.long)
    dirlab = torch.tensor([rg.dir_label[(int(a), int(b))] for a, b in rg.feas],
                          dtype=torch.long)
    return src, dst, dirlab


def _collect_transitions(rg, max_per_t=None):
    """Per-vehicle (t, j, e, e_next) segment transitions from the trace."""
    feas_set = {(int(a), int(b)) for a, b in rg.feas}
    by_t = {}
    for j in range(rg.N):
        prev_e, prev_t = -1, 0
        for t in range(rg.T):
            e = int(rg.edge_idx[j, t])
            if e < 0:
                continue
            if prev_e >= 0 and e != prev_e and (prev_e, e) in feas_set:
                by_t.setdefault(prev_t, []).append((j, prev_e, e))
            if e != prev_e:
                prev_e, prev_t = e, t
    return by_t


def train_hier_gat(rg, epochs=120, lr=5e-3, R=200.0, n_t=40, seed=0,
                   verbose=False, vehicle_specific=True):
    """Train HierGAT on per-vehicle next-segment transitions (Eq. hgat loss)."""
    torch.manual_seed(seed)
    src, dst, dirlab = _feas_tensors(rg)
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(rg.feas)}
    by_t = _collect_transitions(rg)
    times = sorted(by_t.keys())
    if not times:
        if verbose:
            print("  hgat: no transitions found; returning untrained model")
        return HierGAT()
    rng = np.random.RandomState(seed)
    sample_t = times if len(times) <= n_t else \
        sorted(rng.choice(times, n_t, replace=False).tolist())

    model = HierGAT(vehicle_specific=vehicle_specific)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.zeros(())
        ntr = 0
        for t in sample_t:
            hr, hv = model.embed(rg, t, R, src, dst, dirlab)
            o = model.trans_logits(hr, hv, src, dst, dirlab)   # [N,F]
            for j, e, en in by_t[t]:
                row = row_of.get((e, en))
                if row is None:
                    continue
                pi = scatter_softmax(o[j], src, rg.E)
                loss = loss - torch.log(pi[row] + 1e-12)
                ntr += 1
        if ntr == 0:
            break
        loss = loss / ntr
        loss.backward()
        opt.step()
        if verbose and ep % 20 == 0:
            print(f"  hgat epoch {ep}: loss {loss.item():.4f} ({ntr} tr)")
    model.eval()
    return model


@torch.no_grad()
def hier_reachability(model, rg, t, R=200.0, h_max=3, gamma=0.8, mu=1.0):
    """Vehicle-specific reachability Gamma_j [N] (Eq. hgat_road_reachability)."""
    src, dst, dirlab = _feas_tensors(rg)
    hr, hv = model.embed(rg, t, R, src, dst, dirlab)
    o = model.trans_logits(hr, hv, src, dst, dirlab)          # [N,F]
    # per-(vehicle, src) softmax over feasible edges -> pi [N,F]
    Nveh, Fn = o.shape
    flat_src = (torch.arange(Nveh).unsqueeze(1) * rg.E + src.unsqueeze(0)).reshape(-1)
    pi = scatter_softmax(o.reshape(-1), flat_src, Nveh * rg.E).reshape(Nveh, Fn)

    w = rg.window(t)
    qn = rg.q[w] / (rg.q_max + 1e-9)
    weight = torch.tensor(rg.rho[w] * rg.lengths * (1.0 + mu * qn),
                          dtype=torch.float32)              # [E]
    seg = rg.edge_idx[:, t]
    P = torch.zeros(Nveh, rg.E)
    valid = seg >= 0
    P[torch.arange(Nveh)[valid], torch.tensor(seg[valid], dtype=torch.long)] = 1.0
    Gam = torch.zeros(Nveh)
    for h in range(1, h_max + 1):
        contrib = P[:, src] * pi                            # [N,F]
        newP = torch.zeros(Nveh, rg.E)
        newP.index_add_(1, dst, contrib)
        P = newP
        Gam = Gam + (gamma ** h) * (P @ weight)
    return Gam.numpy().astype(np.float32)
