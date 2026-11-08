"""Run with:  python -m pytest -q"""
import itertools
import os
import tempfile

from humanitarian_csp import (Aircraft, CargoItem, FlightSlot, Instance, HumanitarianCSP,
                              BacktrackingSolver, UNASSIGNED, evaluate, diagnose_unassigned,
                              make_demo_instance)


def tiny_instance():
    """6 items, 3 flights, deliberately too small to carry everything."""
    fleet = [
        Aircraft("A", "Big", payload_kg=1000, volume_m3=10, cold_volume_m3=2, turnaround_h=5),
        Aircraft("B", "Small, no cold", payload_kg=400, volume_m3=4, cold_volume_m3=0, turnaround_h=5),
    ]
    slots = [FlightSlot("F1", "A", 10, 14), FlightSlot("F2", "B", 12, 16), FlightSlot("F3", "A", 30, 34)]
    items = [
        CargoItem("P1-a", 300, 1.5, 1, cold_chain=True, ready_h=0, deadline_h=20),     # only F1 (cold + early)
        CargoItem("P1-b", 500, 4, 1, ready_h=0, deadline_h=40),
        CargoItem("P2-c", 350, 3, 2, ready_h=0, deadline_h=40),
        CargoItem("P3-d", 450, 5, 3, ready_h=0, deadline_h=40),
        CargoItem("P2-e", 200, 1, 2, hazmat=True, ready_h=0, deadline_h=40),
        CargoItem("P3-f", 300, 2, 3, category="food", ready_h=0, deadline_h=40),
    ]
    return Instance(items, fleet, slots, name="tiny")


def brute_force_best(csp):
    best = float("inf")
    doms = [csp.initial_domains[i] for i in range(csp.n_vars)]
    for combo in itertools.product(*doms):
        a = csp.new_assignment()
        ok = True
        for i, v in enumerate(combo):
            if a.violation(i, v) is not None:
                ok = False
                break
            a.assign(i, v)
        if ok:
            best = min(best, a.cost)
    return best


def test_solver_matches_brute_force_optimum():
    inst = tiny_instance()
    assert inst.validate() == []
    csp = HumanitarianCSP(inst)
    res = BacktrackingSolver(csp, time_limit_s=None).solve()
    assert res.proven_optimal
    assert abs(res.cost - brute_force_best(csp)) < 1e-6
    assert evaluate(inst, res.plan)["hard_violations_total"] == 0


def test_unary_constraints_prune_domains():
    csp = HumanitarianCSP(tiny_instance())
    # P1-a is cold-chain with deadline 20 -> only F1 (F2 has no cold storage, F3 arrives at 34)
    assert csp.initial_domains[0] == [0, UNASSIGNED]


def test_hazmat_never_shares_flight_with_sensitive_cargo():
    inst = tiny_instance()
    res = BacktrackingSolver(HumanitarianCSP(inst), time_limit_s=None).solve()
    haz, food, cold = res.plan["P2-e"], res.plan["P3-f"], res.plan["P1-a"]
    assert haz is None or (haz != food and haz != cold)


def test_no_violations_on_generated_instances():
    for n, seed in [(20, 1), (40, 2), (60, 3)]:
        for factor in (0.1, 0.8):   # 0.1 = very tight capacity
            inst = make_demo_instance(n, seed, capacity_factor=factor)
            assert inst.validate() == []
            res = BacktrackingSolver(HumanitarianCSP(inst), time_limit_s=1).solve()
            m = evaluate(inst, res.plan)
            assert m["hard_violations_total"] == 0, m["hard_violations"]


def test_evaluator_detects_violations():
    inst = tiny_instance()
    bad = {i.item_id: "F2" for i in inst.items}        # everything on the small no-cold aircraft
    m = evaluate(inst, bad)
    assert m["hard_violations_total"] > 0
    assert "weight_capacity" in m["hard_violations"]
    assert "cold_chain_no_cold_storage" in m["hard_violations"]


def test_unassigned_items_are_explained():
    inst = tiny_instance()
    res = BacktrackingSolver(HumanitarianCSP(inst), time_limit_s=None).solve()
    why = diagnose_unassigned(inst, res.plan)
    assert set(why) == {k for k, v in res.plan.items() if v is None}


def test_first_solution_only_mode_stops_early():
    inst = make_demo_instance(40, 5)
    res = BacktrackingSolver(HumanitarianCSP(inst), first_solution_only=True).solve()
    assert res.stats.solutions_found == 1 and not res.proven_optimal


def test_json_roundtrip():
    inst = make_demo_instance(15, 7)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "i.json")
        inst.save_json(p)
        back = Instance.load_json(p)
    assert back.items == inst.items and back.slots == inst.slots and back.aircraft == inst.aircraft
