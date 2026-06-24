"""CLI experiment runner. Saves one JSON per (method, config, seed)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import run  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--trace", default="med_n60_s1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=55)
    ap.add_argument("--r-v2v", type=float, default=200.0)
    ap.add_argument("--cache-encoders", type=int, default=4)
    ap.add_argument("--max-out", type=int, default=3)
    ap.add_argument("--V", type=float, default=50.0)
    ap.add_argument("--no-gat", action="store_true")
    ap.add_argument("--dataset", default="fmnist",
                    choices=["fmnist", "har", "mmfi", "raymob", "raymob8"])
    ap.add_argument("--phi-agg", type=float, default=0.0,
                    help="staleness discount in aggregation weights")
    ap.add_argument("--local-steps", type=int, default=10,
                    help="local SGD steps per round")
    ap.add_argument("--fusion-mode", default="mean",
                    choices=["mean", "concat"],
                    help="local fusion head: mean (legacy) or concat (slots)")
    ap.add_argument("--starve-frac", type=float, default=0.0,
                    help="fraction of vehicles starved on one owned modality")
    ap.add_argument("--gated-agg", action="store_true",
                    help="learning-gain-gated aggregation (reject negative transfer)")
    ap.add_argument("--share-fusion", action="store_true",
                    help="also FedAvg neighbours' fusion modules each round")
    ap.add_argument("--p-mod", type=float, default=0.72,
                    help="prob each vehicle owns each modality (lower=more missing)")
    ap.add_argument("--q-low-frac", type=float, default=0.35,
                    help="fraction of modalities with degraded sensing quality")
    ap.add_argument("--q-low-hi", type=float, default=0.55,
                    help="upper bound of the low-quality range (lower=harsher)")
    ap.add_argument("--qual-noise", type=float, default=0.8,
                    help="noise multiplier for low-quality modalities (higher=harsher)")
    ap.add_argument("--modality-mode", default=None,
                    help="'complementary' splits fmnist into disjoint noisy thirds")
    ap.add_argument("--comp-noise", type=float, default=None,
                    help="sensor-noise std for complementary modalities")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    cfg = {
        "method": args.method,
        "trace": os.path.join(BASE, "results", "traces",
                              f"trace_{args.trace}.npz"),
        "seed": args.seed,
        "rounds": args.rounds,
        "r_v2v": args.r_v2v,
        "cache_encoders": args.cache_encoders,
        "max_out": args.max_out,
        "V": args.V,
        "use_gat": not args.no_gat,
        "dataset": args.dataset,
        "phi_agg": args.phi_agg,
        "local_steps": args.local_steps,
        "fusion_mode": args.fusion_mode,
        "starve_frac": args.starve_frac,
        "gated_agg": args.gated_agg,
        "share_fusion": args.share_fusion,
        "modality_mode": args.modality_mode,
        "p_mod": args.p_mod,
        "q_low_frac": args.q_low_frac,
        "q_low_range": (0.1, args.q_low_hi),
        "qual_noise": args.qual_noise,
    }
    if args.comp_noise is not None:
        cfg["comp_noise"] = args.comp_noise
    hist = run(cfg)
    out_dir = os.path.join(BASE, "results", "runs")
    os.makedirs(out_dir, exist_ok=True)
    name = (f"{args.tag}_{args.method}_{args.trace}_R{int(args.r_v2v)}"
            f"_C{args.cache_encoders}_V{int(args.V)}_s{args.seed}.json")
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump({"cfg": {k: v for k, v in cfg.items()}, "hist": hist}, f)
    print("saved", name)


if __name__ == "__main__":
    main()
