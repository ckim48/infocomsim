"""Caching and dissemination methods.

RECD (proposed, Sec. III) and three benchmarks:
  - DFL-Gossip      : no caching; vehicles exchange only their own encoders
                      with current neighbors (standard decentralized FL).
  - LRU-Random      : store-carry-forward with LRU cache and random
                      dissemination (utility-agnostic caching baseline).
  - Mobility-Greedy : mobility-aware greedy using instantaneous link
                      expiration time (LET) prediction and learning utility,
                      but no road-graph awareness and no Lyapunov control
                      (represents prior mobility-aware DFL schemes).

A transmission is a tuple (i, j, key) where key = (source_vid, modality).
Senders split bandwidth equally over scheduled links; max_out limits the
number of simultaneous outgoing transmissions per sender.
"""
import numpy as np

from fl import ENC_BITS, N_MOD
import comm


class CacheEntry:
    __slots__ = ("vec", "D", "Q", "birth", "last_used")

    def __init__(self, vec, D, Q, birth):
        self.vec = vec
        self.D = D
        self.Q = Q
        self.birth = birth
        self.last_used = birth


def need_weight(veh, r, eps=1e-3):
    """alpha_need, Eq. (29)."""
    if r not in veh.D:
        return 0.0
    inv = {rr: 1.0 / (veh.D[rr] + eps) for rr in veh.mods}
    return inv[r] / sum(inv.values())


def reliability(veh, r, Dm, Qm, eps=1e-3):
    """beta, Eq. (30)."""
    Di = veh.D.get(r, 0)
    return (Dm * Qm) / (Di + Dm * Qm + eps)


class MethodBase:
    name = "base"
    uses_cache = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg["seed"] * 31 + 7)

    def decide(self, ctx):
        """Return list of transmissions [(i, j, key)]."""
        raise NotImplementedError

    def select_cache(self, veh_id, ctx, candidates, capacity_bits):
        """Default: keep highest-value entries (overridden per method).

        candidates: dict key -> CacheEntry. Returns set of keys to keep.
        """
        raise NotImplementedError

    # helpers -------------------------------------------------------
    def offerings(self, ctx, i):
        """Keys vehicle i can transmit: own encoders + cached encoders."""
        veh = ctx["vehicles"][i]
        keys = [(i, r) for r in veh.mods]
        if self.uses_cache:
            keys += list(ctx["caches"][i].keys())
        return keys

    def meta(self, ctx, i, key):
        """(D, Q, staleness) of encoder `key` held by vehicle i."""
        src, r = key
        if src == i:
            veh = ctx["vehicles"][i]
            return veh.D[r], veh.quality[r], 0
        e = ctx["caches"][i][key]
        return e.D, e.Q, ctx["round"] - e.birth


# ------------------------------------------------------------------ RECD
class RECD(MethodBase):
    name = "RECD"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.V = cfg.get("V", 50.0)
        self.nu = cfg.get("nu", 0.3)
        self.phi = cfg.get("phi", 0.08)
        self.eps_rate = cfg.get("eps_rate", 0.15)
        self.Ebar = cfg.get("Ebar", 3.0)  # J per round budget
        self.Y = None
        self.Z = None

    def decide(self, ctx):
        N = ctx["N"]
        if self.Y is None:
            self.Y = np.zeros((N, N_MOD))
            self.Z = np.zeros(N)
        rg, prm, t = ctx["rg"], ctx["prm"], ctx["t"]
        gamma_seg = ctx["gamma_seg"]  # GAT-predicted reachability per segment
        nbrs, dist = ctx["nbrs"], ctx["dist"]
        vehicles = ctx["vehicles"]

        # per-vehicle predicted reachability Gamma_j
        Gam = np.zeros(N)
        for j in range(N):
            e = rg.edge_idx[j, t]
            Gam[j] = gamma_seg[e] if e >= 0 else 0.0
        Gn = Gam / (Gam.max() + 1e-9)

        # Stage 2 (sender-side max-weight matching, Sec. III-C)
        txs = []
        for i in range(N):
            if not nbrs[i]:
                continue
            cands = []
            for key in self.offerings(ctx, i):
                src, r = key
                D, Q, dl = self.meta(ctx, i, key)
                for j in nbrs[i]:
                    if src == j:
                        continue
                    veh_j = vehicles[j]
                    Tcon = comm.predicted_contact(rg, t, i, j, dist[i, j], prm)
                    rate1 = comm.link_rate(dist[i, j], 1, prm)
                    Ttx1 = ENC_BITS[r] / rate1
                    if Ttx1 > Tcon:  # (C1) infeasible even with full bandwidth
                        continue
                    ptx = np.exp(-Ttx1 / (Tcon + 1e-6))
                    learn = 0.0
                    if r in veh_j.mods:
                        learn = (need_weight(veh_j, r)
                                 * reliability(veh_j, r, D, Q)
                                 * np.exp(-self.phi * dl))
                    w = (self.V * (learn + self.nu * ptx * Gn[j])
                         + self.Y[j, r] * (1.0 if r in veh_j.mods else 0.0)
                         - self.Z[i] * prm.P_watt * Ttx1)
                    if w > 0:
                        cands.append((w, i, j, key, Ttx1, Tcon))
            cands.sort(key=lambda x: -x[0])
            # greedy max-weight selection; re-check (C1) as the bandwidth
            # split grows with each scheduled link
            sel = []
            seen = set()  # (j, modality): C4 one delivery per modality/receiver
            for w, _, j, key, Ttx1, Tcon in cands:
                if len(sel) >= self.cfg["max_out"]:
                    break
                if (j, key[1]) in seen:
                    continue
                n_new = len(sel) + 1
                if all(T1 * n_new <= Tc for _, _, T1, Tc in sel) \
                        and Ttx1 * n_new <= Tcon:
                    sel.append((j, key, Ttx1, Tcon))
                    seen.add((j, key[1]))
            txs.extend((i, j, key) for j, key, _, _ in sel)
        return txs

    def select_cache(self, veh_id, ctx, candidates, capacity_bits):
        """Stage 1: density-greedy knapsack by unified utility (Sec. III-C)."""
        rg, t = ctx["rg"], ctx["t"]
        veh = ctx["vehicles"][veh_id]
        e = rg.edge_idx[veh_id, t]
        g = ctx["gamma_seg"][e] if e >= 0 else 0.0
        gmax = ctx["gamma_seg"].max() + 1e-9
        scored = []
        for key, ent in candidates.items():
            src, r = key
            dl = ctx["round"] - ent.birth
            learn = 0.0
            if r in veh.mods:
                learn = (need_weight(veh, r) * reliability(veh, r, ent.D, ent.Q)
                         * np.exp(-self.phi * dl))
            value = learn + self.nu * (g / gmax) * np.exp(-self.phi * dl)
            scored.append((value / ENC_BITS[r], value, key))
        scored.sort(key=lambda x: -x[0])
        keep, used = set(), 0.0
        for dens, val, key in scored:
            if used + ENC_BITS[key[1]] <= capacity_bits:
                keep.add(key)
                used += ENC_BITS[key[1]]
        return keep

    def update_queues(self, ctx, received_by, energy_by):
        N = ctx["N"]
        for i in range(N):
            got = set(r for (_, r) in received_by[i])
            for r in ctx["vehicles"][i].mods:
                x = 1.0 if r in got else 0.0
                self.Y[i, r] = max(self.Y[i, r] + self.eps_rate - x, 0.0)
            self.Z[i] = max(self.Z[i] + energy_by[i] - self.Ebar, 0.0)


# ------------------------------------------------------------- benchmarks
class DFLGossip(MethodBase):
    name = "DFL-Gossip"
    uses_cache = False

    def decide(self, ctx):
        rg, prm, t = ctx["rg"], ctx["prm"], ctx["t"]
        nbrs, dist = ctx["nbrs"], ctx["dist"]
        txs = []
        for i in range(ctx["N"]):
            veh = ctx["vehicles"][i]
            cands = []
            for j in nbrs[i]:
                veh_j = ctx["vehicles"][j]
                for r in veh.mods:
                    if r in veh_j.mods:
                        cands.append((dist[i, j], i, j, (i, r)))
            cands.sort(key=lambda x: x[0])  # closest neighbors first
            for _, _, j, key in cands[: self.cfg["max_out"]]:
                txs.append((i, j, key))
        return txs

    def select_cache(self, veh_id, ctx, candidates, capacity_bits):
        return set()  # no caching


class LRURandom(MethodBase):
    name = "LRU-Random"

    def decide(self, ctx):
        nbrs = ctx["nbrs"]
        txs = []
        for i in range(ctx["N"]):
            if not nbrs[i]:
                continue
            keys = self.offerings(ctx, i)
            self.rng.shuffle(keys)
            sent = 0
            for key in keys:
                if sent >= self.cfg["max_out"]:
                    break
                j = int(self.rng.choice(nbrs[i]))
                if key[0] == j:
                    continue
                txs.append((i, j, key))
                sent += 1
        return txs

    def select_cache(self, veh_id, ctx, candidates, capacity_bits):
        """LRU eviction: keep most recently received/used entries."""
        items = sorted(candidates.items(), key=lambda kv: -kv[1].last_used)
        keep, used = set(), 0.0
        for key, ent in items:
            if used + ENC_BITS[key[1]] <= capacity_bits:
                keep.add(key)
                used += ENC_BITS[key[1]]
        return keep


class MobilityGreedy(MethodBase):
    """Velocity-based contact prediction (LET), learning utility, greedy.

    Road-agnostic: no road graph, no reachability, no Lyapunov queues.
    """
    name = "Mobility-Greedy"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.phi = cfg.get("phi", 0.08)

    def decide(self, ctx):
        rg, prm, t = ctx["rg"], ctx["prm"], ctx["t"]
        nbrs, dist = ctx["nbrs"], ctx["dist"]
        txs = []
        for i in range(ctx["N"]):
            if not nbrs[i]:
                continue
            cands = []
            for key in self.offerings(ctx, i):
                src, r = key
                D, Q, dl = self.meta(ctx, i, key)
                for j in nbrs[i]:
                    if src == j:
                        continue
                    veh_j = ctx["vehicles"][j]
                    if r not in veh_j.mods:
                        continue
                    let = comm.let_contact(rg, t, i, j, prm)
                    rate1 = comm.link_rate(dist[i, j], 1, prm)
                    Ttx1 = ENC_BITS[r] / rate1
                    if Ttx1 > let:
                        continue
                    w = (need_weight(veh_j, r) * reliability(veh_j, r, D, Q)
                         * np.exp(-self.phi * dl) * np.exp(-Ttx1 / (let + 1e-6)))
                    cands.append((w, i, j, key, Ttx1, let))
            cands.sort(key=lambda x: -x[0])
            sel, seen = [], set()
            for w, _, j, key, Ttx1, let in cands:
                if len(sel) >= self.cfg["max_out"]:
                    break
                if (j, key[1]) in seen:
                    continue
                n_new = len(sel) + 1
                if all(T1 * n_new <= Tc for _, _, T1, Tc in sel) \
                        and Ttx1 * n_new <= let:
                    sel.append((j, key, Ttx1, let))
                    seen.add((j, key[1]))
            txs.extend((i, j, key) for j, key, _, _ in sel)
        return txs

    def select_cache(self, veh_id, ctx, candidates, capacity_bits):
        veh = ctx["vehicles"][veh_id]
        scored = []
        for key, ent in candidates.items():
            src, r = key
            dl = ctx["round"] - ent.birth
            w = 0.0
            if r in veh.mods:
                w = (need_weight(veh, r) * reliability(veh, r, ent.D, ent.Q)
                     * np.exp(-self.phi * dl))
            scored.append((w, key))
        scored.sort(key=lambda x: -x[0])
        keep, used = set(), 0.0
        for w, key in scored:
            if used + ENC_BITS[key[1]] <= capacity_bits:
                keep.add(key)
                used += ENC_BITS[key[1]]
        return keep


METHODS = {
    "RECD": RECD,
    "DFL-Gossip": DFLGossip,
    "LRU-Random": LRURandom,
    "Mobility-Greedy": MobilityGreedy,
}
