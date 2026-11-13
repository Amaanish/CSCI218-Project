"""
Benchmark baseline search against CSP optimization configurations.

Run from the project root:
    python benchmark_optimization.py

The script assumes the package is named `humanitarian_csp` and that
optimization.py is inside that package. It writes benchmark_results.csv.
"""
import csv
import statistics
from pathlib import Path

from humanitarian_csp import HumanitarianCSP, BacktrackingSolver, make_demo_instance
from humanitarian_csp.optimization import (
    mrv_selector,
    degree_selector,
    mrv_degree_selector,
    lcv_value_orderer,
    forward_checking,
    forward_checking_ac3,
)
from humanitarian_csp.solver import static_var_selector, static_value_orderer, no_inference


CONFIGS = [
    ("Baseline", static_var_selector, static_value_orderer, no_inference),
    ("MRV", mrv_selector, static_value_orderer, no_inference),
    ("Degree", degree_selector, static_value_orderer, no_inference),
    ("MRV+Degree+LCV", mrv_degree_selector, lcv_value_orderer, no_inference),
    ("MRV+Degree+LCV+FC", mrv_degree_selector, lcv_value_orderer, forward_checking),
    ("MRV+Degree+LCV+FC+AC3", mrv_degree_selector, lcv_value_orderer, forward_checking_ac3),
]


def main():
    sizes = [50, 100, 200, 500]
    seeds = [1, 2, 3]
    time_limit_s = 3.0
    node_limit = 250_000
    rows = []

    for n_items in sizes:
        for seed in seeds:
            instance = make_demo_instance(n_items=n_items, seed=seed, capacity_factor=0.8)
            for name, selector, orderer, inference in CONFIGS:
                csp = HumanitarianCSP(instance)
                result = BacktrackingSolver(
                    csp,
                    var_selector=selector,
                    value_orderer=orderer,
                    inference=inference,
                    time_limit_s=time_limit_s,
                    node_limit=node_limit,
                ).solve()

                rows.append({
                    "items": n_items,
                    "seed": seed,
                    "configuration": name,
                    "time_ms": round(result.stats.elapsed_s * 1000, 3),
                    "nodes": result.stats.nodes,
                    "backtracks": result.stats.backtracks,
                    "constraint_checks": result.stats.constraint_checks,
                    "solutions_found": result.stats.solutions_found,
                    "cost": round(result.cost, 3),
                    "proven_optimal": result.proven_optimal,
                    "timed_out": result.stats.timed_out,
                })
                print(
                    f"{n_items:3d} items | seed {seed} | {name:25s} "
                    f"{result.stats.elapsed_s * 1000:9.2f} ms | "
                    f"nodes={result.stats.nodes:9,d} | "
                    f"cost={result.cost:12,.1f} | "
                    f"optimal={result.proven_optimal}"
                )

    output = Path("benchmark_results.csv")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\\nSaved {len(rows)} benchmark rows to {output.resolve()}")


if __name__ == "__main__":
    main()
