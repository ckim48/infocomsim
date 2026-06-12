"""GAT-based road-aware mobility predictor (Sec. III-A).

A small graph attention network over the segment-level road graph predicts
the next-direction distribution for each segment, following the directional
representation of NetTraj. Trained offline on segment transitions recorded
in the SUMO trace; produces the transition kernel \\hat{Pi}(k).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from road_graph import N_DIR

HID = 32
EMB = 8


class RoadGAT(nn.Module):
    def __init__(self, in_dim, n_layers=2):
        super().__init__()
        self.dir_emb = nn.Embedding(N_DIR, EMB)
        self.inp = nn.Linear(in_dim, HID)
        self.W = nn.ModuleList([nn.Linear(HID, HID) for _ in range(n_layers)])
        self.att = nn.ModuleList(
            [nn.Linear(2 * HID + EMB, 1) for _ in range(n_layers)]
        )
        self.out = nn.Linear(HID + EMB, 1)

    def forward(self, x, src, dst, dirlab, E):
        """x: [E, in_dim]; (src, dst, dirlab): feasible transition arrays.

        Returns logits over feasible transitions (grouped by src segment).
        """
        h = torch.relu(self.inp(x))
        de = self.dir_emb(dirlab)
        for W, A in zip(self.W, self.att):
            hw = W(h)
            e = F.leaky_relu(A(torch.cat([hw[src], hw[dst], de], dim=1))).squeeze(-1)
            # softmax over outgoing edges of each src (Eq. 24)
            alpha = scatter_softmax(e, src, E)
            agg = torch.zeros_like(hw)
            agg.index_add_(0, src, alpha.unsqueeze(1) * hw[dst])
            h = torch.relu(hw + agg)
        logits = self.out(torch.cat([h[src], de], dim=1)).squeeze(-1)
        return logits


def scatter_softmax(e, src, E):
    mx = torch.full((E,), -1e30, device=e.device)
    mx = mx.scatter_reduce(0, src, e, reduce="amax")
    ex = torch.exp(e - mx[src])
    den = torch.zeros(E, device=e.device)
    den.index_add_(0, src, ex)
    return ex / (den[src] + 1e-12)


def train_gat(rg, epochs=300, lr=5e-3, seed=0, verbose=False):
    """Train on transition counts from the trace; returns the model."""
    torch.manual_seed(seed)
    src = torch.tensor(rg.feas[:, 0], dtype=torch.long)
    dst = torch.tensor(rg.feas[:, 1], dtype=torch.long)
    dirlab = torch.tensor(
        [rg.dir_label[(a, b)] for a, b in rg.feas], dtype=torch.long
    )
    # empirical next-segment counts (smoothed)
    cnt = torch.full((len(rg.feas),), 0.1)
    key2row = {tuple(k): i for i, k in enumerate(map(tuple, rg.feas))}
    for k, v in zip(map(tuple, rg.trans_keys), rg.trans_vals):
        if k in key2row:
            cnt[key2row[k]] += float(v)
    target = cnt / torch.zeros(rg.E).index_add_(0, src, cnt)[src]

    x = torch.tensor(rg.features(rg.T // 2))
    model = RoadGAT(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(x, src, dst, dirlab, rg.E)
        p = scatter_softmax(logits, src, rg.E)
        loss = -(target * torch.log(p + 1e-12)).sum() / rg.E
        loss.backward()
        opt.step()
        if verbose and ep % 50 == 0:
            print(f"  gat epoch {ep}: loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def gat_kernel(model, rg, t):
    """Predicted transition kernel \\hat{Pi}(k) (Eq. 25) as a dense matrix."""
    src = torch.tensor(rg.feas[:, 0], dtype=torch.long)
    dst = torch.tensor(rg.feas[:, 1], dtype=torch.long)
    dirlab = torch.tensor(
        [rg.dir_label[(a, b)] for a, b in rg.feas], dtype=torch.long
    )
    x = torch.tensor(rg.features(t))
    logits = model(x, src, dst, dirlab, rg.E)
    p = scatter_softmax(logits, src, rg.E).numpy()
    P = np.zeros((rg.E, rg.E), dtype=np.float32)
    P[rg.feas[:, 0], rg.feas[:, 1]] = p
    # absorbing self-loop for segments without outgoing transitions
    rs = P.sum(1)
    P[rs < 1e-6, :] = 0.0
    P[np.where(rs < 1e-6)[0], np.where(rs < 1e-6)[0]] = 1.0
    return P
