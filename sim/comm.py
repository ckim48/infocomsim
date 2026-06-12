"""V2V communication model (Sec. II-C).

Orthogonal bandwidth sharing, log-distance path loss, contact-duration
prediction from segment speeds (Eqs. 11-13), and ground-truth transmission
success evaluated against the recorded trace: a transmission of duration
T_tx started at time t succeeds iff the pair stays within R_V2V for the
whole interval [t, t+T_tx].
"""
import numpy as np


class V2VParams:
    def __init__(self):
        self.R = 300.0            # V2V range (m)
        self.B = 10e6             # total bandwidth (Hz)
        self.P_dbm = 23.0         # transmit power
        self.noise_dbm = -95.0    # noise power
        self.pl0_db = 45.0        # path loss at 1 m
        self.alpha = 3.0          # path loss exponent
        self.eps = 1e-6

    @property
    def P_watt(self):
        return 10 ** ((self.P_dbm - 30) / 10)


def distances(rg, t):
    """Pairwise distances at time t; NaN-safe. Returns [N, N]."""
    p = rg.pos[:, min(t, rg.T - 1)]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    bad = ~np.isfinite(p[:, 0])
    d[bad, :] = np.inf
    d[:, bad] = np.inf
    np.fill_diagonal(d, np.inf)
    return d


def neighbors(rg, t, prm):
    d = distances(rg, t)
    return [list(np.where(d[i] <= prm.R)[0]) for i in range(rg.N)], d


def link_rate(d, n_links, prm):
    """Achievable rate (bit/s) when the sender splits bandwidth over n links."""
    d = max(d, 1.0)
    pl_db = prm.pl0_db + 10 * prm.alpha * np.log10(d)
    snr_db = prm.P_dbm - pl_db - prm.noise_dbm
    snr = 10 ** (snr_db / 10)
    b = prm.B / max(n_links, 1)
    return b * np.log2(1 + snr)


def predicted_contact(rg, t, i, j, d_ij, prm):
    """Conservative contact-duration estimate, Eqs. (11)-(13)."""
    ei, ej = rg.edge_idx[i, t], rg.edge_idx[j, t]
    w = rg.window(t)
    vi = rg.vbar[w, ei] if ei >= 0 else 8.0
    vj = rg.vbar[w, ej] if ej >= 0 else 8.0
    # heading sign via instantaneous velocity projection onto segment heading
    si = _sign(rg, t, i, ei)
    sj = _sign(rg, t, j, ej)
    rel = abs(vi * si - vj * sj) + 0.5
    t_rng = max(prm.R - d_ij, 0.0) / rel
    tau_i = _residual(rg, t, i, ei, vi)
    tau_j = _residual(rg, t, j, ej, vj)
    return max(min(t_rng, tau_i, tau_j), 0.5)


def _sign(rg, t, i, e):
    if e < 0 or t + 1 >= rg.T:
        return 1
    dp = rg.pos[i, min(t + 1, rg.T - 1)] - rg.pos[i, t]
    if not np.all(np.isfinite(dp)):
        return 1
    hd = np.radians(rg.headings[e])
    return 1 if dp @ np.array([np.cos(hd), np.sin(hd)]) >= 0 else -1


def _residual(rg, t, i, e, v):
    if e < 0:
        return 5.0
    rem = max(rg.lengths[e] - rg.lane_pos[i, t], 0.0)
    return rem / (v + 1e-3)


def actual_contact_remaining(rg, t, i, j, prm):
    """Ground truth: seconds the pair remains within R from time t on."""
    tt = t
    while tt < rg.T:
        p1, p2 = rg.pos[i, tt], rg.pos[j, tt]
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            break
        if np.linalg.norm(p1 - p2) > prm.R:
            break
        tt += 1
    return tt - t


def let_contact(rg, t, i, j, prm):
    """Link expiration time from instantaneous positions/velocities only
    (classic LET; used by the road-agnostic Mobility-Greedy baseline)."""
    if t + 1 >= rg.T:
        return 1.0
    p = rg.pos[:, t]
    v = rg.pos[:, min(t + 1, rg.T - 1)] - p
    if not (np.all(np.isfinite(p[i])) and np.all(np.isfinite(p[j]))
            and np.all(np.isfinite(v[i])) and np.all(np.isfinite(v[j]))):
        return 1.0
    dp = p[i] - p[j]
    dv = v[i] - v[j]
    a = dv @ dv
    b = 2 * dp @ dv
    c = dp @ dp - prm.R ** 2
    if a < 1e-9:
        return 1e4 if c <= 0 else 1.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return 1.0
    tau = (-b + np.sqrt(disc)) / (2 * a)
    return max(tau, 0.5)
