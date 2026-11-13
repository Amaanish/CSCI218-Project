"""
run_experiments.py -- Repeated trials: CSP backtracking vs greedy baselines.

Usage (from the repo root):
    python experiments/run_experiments.py --trials 2 --time-limit 2     # quick test
    python experiments/run_experiments.py                                # full run
Output: experiments/results/raw_results.csv  (one row per algorithm per trial)
"""
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))        # so "humanitarian_csp" can be imported

from humanitarian_csp import BacktrackingSolver, HumanitarianCSP, evaluate, make_demo_instance
from greedy_baseline import greedy_feasible, greedy_naive

SIZES = {"small": 50, "medium": 200, "large": 500}


# ---- each algorithm takes an Instance and returns (plan, extra_info_dict) -------------
def run_csp_baseline(instance, time_limit):
    csp = HumanitarianCSP(instance)
    res = BacktrackingSolver(csp, time_limit_s=time_limit).solve()
    return res.plan, {
        "timed_out": res.stats.timed_out,
        "proven_optimal": res.proven_optimal,
        "nodes": res.stats.nodes,
        "backtracks": res.stats.backtracks,
        "constraint_checks": res.stats.constraint_checks,
    }


def run_greedy_naive(instance, time_limit):
    return greedy_naive(instance), {}


def run_greedy_feasible(instance, time_limit):
    return greedy_feasible(instance), {}


ALGORITHMS = {
    "CSP-Backtracking": run_csp_baseline,
    "Greedy-Naive": run_greedy_naive,
    "Greedy-Feasible": run_greedy_feasible,
    # LATER: add the heuristic versions once optimization.py is reviewed, e.g.
    # "CSP+MRV+FC": run_csp_with_heuristics,
}

FIELDS = ["size", "n_items", "trial", "seed", "algorithm", "time_ms", "total_cost",
          "hard_violations", "unmet_items_pct", "unmet_weight_pct",
          "p1_served_pct", "p2_served_pct", "p3_served_pct",
          "mean_wait_h", "flights_used", "timed_out", "proven_optimal",
          "nodes", "backtracks", "constraint_checks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--time-limit", type=float, default=10.0, help="CSP time limit (seconds)")
    ap.add_argument("--capacity", type=float, default=0.8, help="airlift capacity / demand")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_results.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for size_name, n in SIZES.items():
            for trial in range(args.trials):
                seed = trial                      # same seed -> every algorithm sees the SAME instance
                instance = make_demo_instance(n_items=n, seed=seed, capacity_factor=args.capacity)
                for algo_name, fn in ALGORITHMS.items():
                    t0 = time.perf_counter()
                    plan, extra = fn(instance, args.time_limit)
                    time_ms = (time.perf_counter() - t0) * 1000.0
                    m = evaluate(instance, plan)          # independent, same checker for everyone
                    ps = m["priority_satisfaction"]
                    row = {
                        "size": size_name, "n_items": n, "trial": trial, "seed": seed,
                        "algorithm": algo_name, "time_ms": round(time_ms, 2),
                        "total_cost": round(m["total_cost"], 2),
                        "hard_violations": m["hard_violations_total"],
                        "unmet_items_pct": round(m["unmet_demand_pct_items"], 2),
                        "unmet_weight_pct": round(m["unmet_demand_pct_weight"], 2),
                        "p1_served_pct": ps.get("P1"), "p2_served_pct": ps.get("P2"),
                        "p3_served_pct": ps.get("P3"),
                        "mean_wait_h": m["mean_delivery_wait_h"],
                        "flights_used": m["flights_used"],
                        "timed_out": extra.get("timed_out", False),
                        "proven_optimal": extra.get("proven_optimal", ""),
                        "nodes": extra.get("nodes", ""),
                        "backtracks": extra.get("backtracks", ""),
                        "constraint_checks": extra.get("constraint_checks", ""),
                    }
                    writer.writerow(row)
                    f.flush()
                print(f"[{size_name}] trial {trial + 1}/{args.trials} done")
    print("Saved", out_path)


if __name__ == "__main__":
    main()