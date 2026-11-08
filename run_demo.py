"""
Demo:  python run_demo.py [n_items] [seed] [capacity_factor] [time_limit_s]
Builds a demo instance, solves it with the baseline backtracking solver, prints the schedule.
"""
import sys
from collections import defaultdict

from humanitarian_csp import (HumanitarianCSP, BacktrackingSolver, evaluate,
                              diagnose_unassigned, make_demo_instance)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    factor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15      # tight -> forces trade-offs
    limit = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

    inst = make_demo_instance(n, seed, capacity_factor=factor)
    csp = HumanitarianCSP(inst)
    print(f"Instance {inst.name}: {len(inst.items)} items, {len(inst.slots)} flight slots, "
          f"{sum(len(d) - 1 for d in csp.initial_domains.values())} feasible (item, slot) pairs")

    res = BacktrackingSolver(csp, time_limit_s=limit).solve()
    st = res.stats
    print(f"\nBaseline backtracking: cost={res.cost:,.1f}  proven_optimal={res.proven_optimal}  "
          f"nodes={st.nodes:,}  backtracks={st.backtracks:,}  time={st.elapsed_s:.2f}s")

    by_slot = defaultdict(list)
    for iid, sid in res.plan.items():
        if sid:
            by_slot[sid].append(iid)
    ac = inst.aircraft_by_id()
    print("\nSchedule")
    for sl in inst.slots:
        cargo = [it for it in inst.items if it.item_id in by_slot[sl.slot_id]]
        plane = ac[sl.aircraft_id]
        kg = sum(c.weight_kg for c in cargo)
        print(f"  {sl.slot_id} {plane.model:<17} dep {sl.departure_h:5.1f}h -> arr {sl.arrival_h:5.1f}h | "
              f"{len(cargo):2d} items, {kg:7.0f}/{plane.payload_kg:.0f} kg | {', '.join(c.item_id for c in cargo)}")

    m = evaluate(inst, res.plan)
    print("\nMetrics")
    print(f"  hard-constraint violations : {m['hard_violations_total']}")
    print(f"  unmet demand               : {m['unmet_demand_pct_items']:.1f}% of items, {m['unmet_demand_pct_weight']:.1f}% of weight")
    print(f"  priority satisfaction      : " + ", ".join(f"{k}={v:.0f}%" for k, v in m['priority_satisfaction'].items()))

    why = diagnose_unassigned(inst, res.plan)
    if why:
        print(f"\nLeft behind ({len(why)}):")
        for iid, reason in list(why.items())[:10]:
            print(f"  {iid}: {reason}")


if __name__ == "__main__":
    main()
