"""Road graph G_road built from a recorded SUMO trace file.

Nodes are directed road segments; edges are feasible intersection
transitions (Sec. II-A of the paper). Per-window traffic attributes
z_e(t) = [L_e, rho_e(t), vbar_e(t), q_e(t)] are taken from the SUMO run.
"""
import numpy as np

N_DIR = 8  # direction labels (Eq. 7)


class RoadGraph:
    def __init__(self, trace_file):
        d = np.load(trace_file, allow_pickle=True)
        self.pos = d["pos"]            # [N, T, 2] participant positions (m)
        self.speed = d["speed"]        # [N, T]
        self.edge_idx = d["edge_idx"]  # [N, T] segment index (-1 unknown)
        self.lane_pos = d["lane_pos"]  # [N, T] distance along segment
        self.lengths = d["lengths"]    # [E]
        self.headings = d["headings"]  # [E] degrees
        self.centers = d["centers"]    # [E, 2]
        self.feas = d["feas"]          # [n_trans, 2] feasible transitions
        self.trans_keys = d["trans_keys"]
        self.trans_vals = d["trans_vals"]
        self.W = int(d["stat_window"][0])
        ec, es, ef = d["edge_count"], d["edge_speed"], d["edge_flow"]
        self.E = len(self.lengths)
        self.N, self.T = self.pos.shape[:2]

        # per-window traffic attributes
        self.rho = ec / (self.lengths[None, :] * self.W)          # veh/m
        with np.errstate(invalid="ignore", divide="ignore"):
            vbar = np.where(ec > 0, es / np.maximum(ec, 1e-9), np.nan)
        # fall back to network-wide mean speed where no observation
        fill = np.nanmean(vbar) if np.isfinite(np.nanmean(vbar)) else 8.0
        self.vbar = np.where(np.isfinite(vbar), vbar, fill)       # m/s
        self.q = ef / self.W                                      # veh/s
        self.q_max = max(self.q.max(), 1e-9)
        self.n_win = self.rho.shape[0]

        # adjacency (feasible transitions) and direction labels (Eq. 7)
        self.out_edges = [[] for _ in range(self.E)]
        for a, b in self.feas:
            self.out_edges[a].append(b)
        wdt = 360.0 / N_DIR
        self.dir_label = {}
        for a, b in self.feas:
            self.dir_label[(a, b)] = int(
                ((self.headings[b] - self.headings[a]) % 360.0) // wdt
            )

    def window(self, t):
        return min(int(t) // self.W, self.n_win - 1)

    def features(self, t):
        """Per-segment features z_e(t) plus heading, normalized."""
        w = self.window(t)
        return np.stack(
            [
                self.lengths / max(self.lengths.max(), 1.0),
                self.rho[w] / max(self.rho.max(), 1e-9),
                self.vbar[w] / max(self.vbar.max(), 1e-9),
                self.q[w] / self.q_max,
                np.sin(np.radians(self.headings)),
                np.cos(np.radians(self.headings)),
            ],
            axis=1,
        ).astype(np.float32)


def markov_kernel(rg, t, psi=None, omega=1.0):
    """Parametric transition kernel Pi(k), Eq. (8).

    psi: directional preference per label (length N_DIR), calibrated from
    historical transition counts; uniform if None.
    """
    if psi is None:
        psi = np.zeros(N_DIR)
    w = rg.window(t)
    qn = rg.q[w] / (rg.q_max + 1e-9)
    P = np.zeros((rg.E, rg.E), dtype=np.float32)
    for e in range(rg.E):
        outs = rg.out_edges[e]
        if not outs:
            P[e, e] = 1.0
            continue
        logits = np.array(
            [psi[rg.dir_label[(e, ep)]] + omega * qn[ep] for ep in outs]
        )
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        for ep, pe in zip(outs, p):
            P[e, ep] = pe
    return P


def calibrate_psi(rg):
    """Directional preference psi_delta from historical transition counts."""
    counts = np.zeros(N_DIR)
    for (a, b), v in zip(map(tuple, rg.trans_keys), rg.trans_vals):
        lbl = rg.dir_label.get((a, b))
        if lbl is not None:
            counts[lbl] += v
    p = (counts + 1.0) / (counts.sum() + N_DIR)
    return np.log(p)


def reachability(rg, t, Pi, h_max=3, gamma=0.8, mu=1.0):
    """Road-aware reachability Gamma_j for every segment, Eq. (16).

    Returns gamma_seg[E]: expected discounted number of encountered vehicles
    for a vehicle currently on each segment.
    """
    w = rg.window(t)
    qn = rg.q[w] / (rg.q_max + 1e-9)
    weight = rg.rho[w] * rg.lengths * (1.0 + mu * qn)  # expected veh per segment
    gamma_seg = np.zeros(rg.E, dtype=np.float32)
    M = np.eye(rg.E, dtype=np.float32)
    for h in range(1, h_max + 1):
        M = M @ Pi
        gamma_seg += (gamma ** h) * (M @ weight)
    return gamma_seg
