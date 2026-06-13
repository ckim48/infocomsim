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
    ap.add_argument("--dataset", default="fmnist", choices=["fmnist", "har"])
    ap.add_argument("--phi-agg", type=float, default=0.0,
                    help="staleness discount in aggregation weights")
    ap.add_argument("--local-steps", type=int, default=10,
                    help="local SGD steps per round")
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
    }
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
