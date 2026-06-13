"""Round-based DFL simulation engine.

Per round k (round length = round_sec seconds of the SUMO trace):
 1. neighbor discovery from real positions
 2. method decides transmissions (i, j, key)
 3. PHY: bandwidth split per sender; T_tx from rate; the transmission
    succeeds iff the pair actually stays within range for T_tx seconds in
    the recorded trace (ground truth) -- mispredicted contacts waste energy
 4. receivers aggregate received encoders for their own modalities (Eq. 20)
    and update their caches via the method's caching rule (C2)
 5. all vehicles run local training (Eqs. 21-22); staleness advances
"""
import time
import numpy as np
import torch

from fl import (Vehicle, modality_views, load_fashion_mnist, partition,
                load_uci_har, har_views, partition_har, assign_heterogeneity,
                SPECS, ENC_BITS, N_MOD, get_vec)
from methods import METHODS, CacheEntry
from road_graph import RoadGraph, markov_kernel, calibrate_psi, reachability
import comm


def jain(x):
    x = np.asarray(x, dtype=float)
    if x.sum() <= 0:
        return 1.0
    return x.sum() ** 2 / (len(x) * (x ** 2).sum() + 1e-12)


def run(cfg):
    t0 = time.time()
    rng = np.random.RandomState(cfg["seed"])
    rg = RoadGraph(cfg["trace"])
    prm = comm.V2VParams()
    prm.R = cfg.get("r_v2v", 300.0)
    N = rg.N

    # ---------------- data & vehicles
    dataset = cfg.get("dataset", "fmnist")
    spec_key = dataset
    if dataset == "fmnist" and cfg.get("modality_mode") == "complementary":
        spec_key = "fmnist_comp"
        import fl
        fl.COMP_NOISE = cfg.get("comp_noise", fl.COMP_NOISE)
    spec = SPECS[spec_key]
    if dataset == "har":
        xtr, ytr, subj_tr, xte, yte = load_uci_har()
        veh_idx = partition_har(subj_tr, N, cfg["seed"])
        mods, mod_frac, quality = assign_heterogeneity(
            N, np.random.RandomState(cfg["seed"]),
            starve_frac=cfg.get("starve_frac", 0.0))
    else:
        xtr, ytr, xte, yte = load_fashion_mnist()
        veh_idx, mods, mod_frac, quality = partition(
            ytr, N, cfg["seed"], starve_frac=cfg.get("starve_frac", 0.0))
    vehicles = [
        Vehicle(i, veh_idx[i], mods[i], mod_frac[i], quality[i], xtr, ytr,
                cfg["seed"], spec=spec,
                fusion_mode=cfg.get("fusion_mode", "mean"))
        for i in range(N)
    ]
    n_test = min(cfg.get("n_test", 1000), len(xte))
    sel = rng.choice(len(xte), n_test, replace=False)
    xte_views = spec["views"](xte[sel])
    yte_s = yte[sel]

    # ---------------- mobility prediction
    use_gat = cfg.get("use_gat", True) and cfg["method"] == "RECD"
    if use_gat:
        import gat_predictor
        gat = gat_predictor.train_gat(rg, epochs=cfg.get("gat_epochs", 250))
    psi = calibrate_psi(rg)

    method = METHODS[cfg["method"]](cfg)
    caches = [dict() for _ in range(N)]
    cap_bits = cfg.get("cache_encoders", 4) * np.mean(list(ENC_BITS.values()))

    K = cfg.get("rounds", 55)
    round_sec = cfg.get("round_sec", 20)
    eval_every = cfg.get("eval_every", 5)

    hist = {
        "acc": [], "acc_round": [], "recv": [], "succ": [], "att": [],
        "energy": [], "coverage": [], "staleness": [], "fail": [],
        "recv_per_veh": np.zeros(N), "first_recv": {},
        "y_queue": [], "z_queue": [],
    }
    covered = set()   # (i, r) pairs that received a foreign encoder
    need_pairs = [(i, r) for i in range(N) for r in vehicles[i].mods]
    cum_energy = np.zeros(N)

    for k in range(K):
        t = min(k * round_sec, rg.T - 2)
        nbrs, dist = comm.neighbors(rg, t, prm)

        # reachability for RECD (GAT kernel; Markov fallback otherwise)
        gamma_seg = None
        if cfg["method"] == "RECD":
            if use_gat:
                import gat_predictor
                Pi = gat_predictor.gat_kernel(gat, rg, t)
            else:
                Pi = markov_kernel(rg, t, psi)
            gamma_seg = reachability(rg, t, Pi, h_max=cfg.get("h_max", 3),
                                     gamma=cfg.get("gamma_disc", 0.8))

        ctx = {
            "rg": rg, "prm": prm, "t": t, "N": N, "round": k,
            "vehicles": vehicles, "caches": caches,
            "nbrs": nbrs, "dist": dist, "gamma_seg": gamma_seg,
        }

        txs = method.decide(ctx)

        # ---------------- PHY simulation
        out_cnt = np.zeros(N, dtype=int)
        for i, j, key in txs:
            out_cnt[i] += 1
        received = [dict() for _ in range(N)]   # j -> key -> (vec, D, Q, dl)
        received_by = [set() for _ in range(N)]
        energy_by = np.zeros(N)
        succ = att = fail = 0
        stale_sum = stale_n = 0
        for i, j, key in txs:
            src, r = key
            rate = comm.link_rate(dist[i, j], out_cnt[i], prm)
            Ttx = ENC_BITS[r] / rate
            real = comm.actual_contact_remaining(rg, t, i, j, prm)
            att += 1
            if Ttx <= real:
                # deliver
                if src == i:
                    vec = get_vec(vehicles[i].enc[r]).cpu()
                    D, Q, dl = vehicles[i].D[r], vehicles[i].quality[r], 0
                else:
                    ent = caches[i].get(key)
                    if ent is None:
                        continue
                    vec, D, Q = ent.vec, ent.D, ent.Q
                    dl = k - ent.birth
                    ent.last_used = k
                cur = received[j].get(key)
                if cur is None or cur[3] > dl:
                    received[j][key] = (vec, D, Q, dl)
                received_by[j].add((src, r))
                energy_by[i] += prm.P_watt * Ttx
                succ += 1
                stale_sum += dl
                stale_n += 1
            else:
                fail += 1
                energy_by[i] += prm.P_watt * real  # wasted energy

        cum_energy += energy_by

        # ---------------- aggregation (Eq. 20) + coverage stats
        phi_agg = cfg.get("phi_agg", 0.0)
        for j in range(N):
            byr = {}
            for (src, r), (vec, D, Q, dl) in received[j].items():
                if r in vehicles[j].mods:
                    byr.setdefault(r, []).append((vec, D, Q, dl))
                    if (j, r) not in hist["first_recv"]:
                        hist["first_recv"][(j, r)] = k
                    covered.add((j, r))
            for r, recvs in byr.items():
                vehicles[j].aggregate(r, recvs, phi_agg=phi_agg,
                                      gated=cfg.get("gated_agg", False))
                hist["recv_per_veh"][j] += len(recvs)

        # ---------------- caching (C2)
        if method.uses_cache:
            for j in range(N):
                cands = dict(caches[j])
                for (src, r), (vec, D, Q, dl) in received[j].items():
                    if src == j:
                        continue
                    ent = CacheEntry(vec, D, Q, k - dl)
                    ent.last_used = k
                    cands[(src, r)] = ent
                keep = method.select_cache(j, ctx, cands, cap_bits)
                caches[j] = {key: cands[key] for key in keep}

        # ---------------- Lyapunov queues (RECD)
        if hasattr(method, "update_queues"):
            method.update_queues(ctx, received_by, energy_by)
            hist["y_queue"].append(float(method.Y.mean()))
            hist["z_queue"].append(float(method.Z.mean()))

        # ---------------- local training
        for v in vehicles:
            v.local_train(steps=cfg.get("local_steps", 10))

        hist["recv"].append(int(succ))
        hist["succ"].append(int(succ))
        hist["att"].append(int(att))
        hist["fail"].append(int(fail))
        hist["energy"].append(float(energy_by.sum()))
        hist["coverage"].append(len(covered) / max(len(need_pairs), 1))
        hist["staleness"].append(stale_sum / max(stale_n, 1))

        if (k + 1) % eval_every == 0 or k == 0:
            accs = [v.evaluate(xte_views, yte_s) for v in vehicles]
            hist["acc"].append(float(np.mean(accs)))
            hist["acc_round"].append(k + 1)
            print(f"[{cfg['method']}] round {k+1}/{K} "
                  f"acc={np.mean(accs):.4f} cov={hist['coverage'][-1]:.3f} "
                  f"succ={succ}/{att} ({time.time()-t0:.0f}s)", flush=True)

    # per-vehicle final accuracy + heterogeneity metadata for breakdowns
    hist["acc_per_veh"] = [v.evaluate(xte_views, yte_s) for v in vehicles]
    hist["veh_meta"] = [
        {"mods": v.mods, "D": {str(r): int(d) for r, d in v.D.items()},
         "Q": {str(r): float(q) for r, q in v.quality.items()}}
        for v in vehicles
    ]

    # final summary metrics
    fr = list(hist["first_recv"].values())
    hist["mean_first_recv"] = float(np.mean(fr)) if fr else float(K)
    hist["jain"] = jain(hist["recv_per_veh"])
    hist["recv_per_veh"] = hist["recv_per_veh"].tolist()
    hist["first_recv"] = {f"{i}_{r}": v for (i, r), v in hist["first_recv"].items()}
    hist["cum_energy"] = cum_energy.tolist()
    hist["wall_sec"] = time.time() - t0
    return hist
