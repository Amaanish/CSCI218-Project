"""
evaluate.py -- Independent checker + metrics for ANY plan (CSP solver, greedy baseline, ...).

A plan is a dict {item_id: slot_id or None}.  Nothing here reuses the solver's incremental
bookkeeping: everything is recomputed from scratch, so it can be trusted as ground truth
when counting constraint violations in the experiments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .model import Instance

_EPS = 1e-9
Plan = Dict[str, Optional[str]]


def evaluate(instance: Instance, plan: Plan) -> dict:
    items = {it.item_id: it for it in instance.items}
    slots = {s.slot_id: s for s in instance.slots}
    ac = instance.aircraft_by_id()

    violations: Counter = Counter()
    on_slot = defaultdict(list)
    for iid, sid in plan.items():
        if sid is None:
            continue
        if sid not in slots or iid not in items:
            violations["unknown_id"] += 1
            continue
        on_slot[sid].append(items[iid])

    # ---- per-item (unary) hard constraints
    for sid, cargo in on_slot.items():
        sl, plane = slots[sid], ac[slots[sid].aircraft_id]
        for it in cargo:
            if it.ready_h + it.handling_h > sl.departure_h + _EPS:
                violations["time_window_ready"] += 1
            if sl.arrival_h > it.deadline_h + _EPS:
                violations["time_window_deadline"] += 1
            if it.cold_chain and plane.cold_volume_m3 <= 0:
                violations["cold_chain_no_cold_storage"] += 1

        # ---- per-slot capacity constraints
        if sum(i.weight_kg for i in cargo) > plane.payload_kg + _EPS:
            violations["weight_capacity"] += 1
        if sum(i.volume_m3 for i in cargo) > plane.volume_m3 + _EPS:
            violations["volume_capacity"] += 1
        if sum(i.volume_m3 for i in cargo if i.cold_chain) > plane.cold_volume_m3 + _EPS:
            violations["cold_storage_capacity"] += 1

        # ---- hazmat segregation
        has_haz = any(i.hazmat for i in cargo)
        has_sens = any(i.cold_chain or i.category == "food" for i in cargo)
        if has_haz and has_sens:
            # count a violation only if a hazmat item is actually paired with a different sensitive item
            for h in (i for i in cargo if i.hazmat):
                if any((o.cold_chain or o.category == "food") and o is not h for o in cargo):
                    violations["hazmat_segregation"] += 1
                    break

    # ---- objective and service metrics
    total_cost, delays = 0.0, []
    tiers = defaultdict(lambda: {"n": 0, "served": 0, "kg": 0.0, "kg_served": 0.0})
    for iid, it in items.items():
        sid = plan.get(iid)
        t = tiers[it.priority]
        t["n"] += 1
        t["kg"] += it.weight_kg
        if sid is None or sid not in slots:
            total_cost += instance.unmet_cost(it)
        else:
            total_cost += instance.delay_cost(it, slots[sid])
            delays.append(slots[sid].arrival_h - it.ready_h)
            t["served"] += 1
            t["kg_served"] += it.weight_kg

    n_items = len(items)
    n_served = sum(t["served"] for t in tiers.values())
    kg_all = sum(t["kg"] for t in tiers.values())
    kg_served = sum(t["kg_served"] for t in tiers.values())

    return {
        "hard_violations_total": sum(violations.values()),
        "hard_violations": dict(violations),
        "total_cost": total_cost,
        "items_total": n_items,
        "items_served": n_served,
        "unmet_demand_pct_items": 100.0 * (n_items - n_served) / n_items if n_items else 0.0,
        "unmet_demand_pct_weight": 100.0 * (kg_all - kg_served) / kg_all if kg_all else 0.0,
        "priority_satisfaction": {                       # % of each tier's items that were shipped
            f"P{p}": (100.0 * t["served"] / t["n"] if t["n"] else None)
            for p, t in sorted(tiers.items())
        },
        "priority_satisfaction_weight": {                # % of each tier's tonnage that was shipped
            f"P{p}": (100.0 * t["kg_served"] / t["kg"] if t["kg"] else None)
            for p, t in sorted(tiers.items())
        },
        "mean_delivery_wait_h": (sum(delays) / len(delays)) if delays else None,
        "flights_used": len(on_slot),
        "flights_available": len(slots),
    }


def diagnose_unassigned(instance: Instance, plan: Plan) -> Dict[str, str]:
    """
    For every item left behind, explain WHY it could not be added to the finished plan.
    This is the raw material for the "unmet demand / constraint conflict" error log:
    e.g.  'P1-007': 'cold storage full on all 3 time-feasible flights'.
    """
    items = {it.item_id: it for it in instance.items}
    ac = instance.aircraft_by_id()
    load = defaultdict(lambda: {"w": 0.0, "v": 0.0, "cold": 0.0, "haz": 0, "sens": 0})
    for iid, sid in plan.items():
        if sid is None:
            continue
        it, L = items[iid], load[sid]
        L["w"] += it.weight_kg
        L["v"] += it.volume_m3
        L["cold"] += it.volume_m3 if it.cold_chain else 0.0
        L["haz"] += int(it.hazmat)
        L["sens"] += int(it.cold_chain or it.category == "food")

    out: Dict[str, str] = {}
    for iid, sid in plan.items():
        if sid is not None:
            continue
        it = items[iid]
        sensitive = it.cold_chain or it.category == "food"
        reasons: Counter = Counter()
        feasible_in_time = 0
        for sl in instance.slots:
            plane = ac[sl.aircraft_id]
            if it.ready_h + it.handling_h > sl.departure_h + _EPS or sl.arrival_h > it.deadline_h + _EPS:
                continue
            feasible_in_time += 1
            L = load[sl.slot_id]
            if it.cold_chain and it.volume_m3 > plane.cold_volume_m3 + _EPS:
                reasons["aircraft has no suitable cold storage"] += 1
            elif it.weight_kg > plane.payload_kg + _EPS or it.volume_m3 > plane.volume_m3 + _EPS:
                reasons["too large for aircraft"] += 1
            elif L["w"] + it.weight_kg > plane.payload_kg + _EPS:
                reasons["weight capacity full"] += 1
            elif L["v"] + it.volume_m3 > plane.volume_m3 + _EPS:
                reasons["volume capacity full"] += 1
            elif it.cold_chain and L["cold"] + it.volume_m3 > plane.cold_volume_m3 + _EPS:
                reasons["cold storage full"] += 1
            elif (it.hazmat and L["sens"] > 0) or (sensitive and L["haz"] > 0):
                reasons["hazmat segregation conflict"] += 1
            else:
                reasons["could still fit (cost/search trade-off)"] += 1
        if feasible_in_time == 0:
            out[iid] = "no flight inside the item's time window"
        else:
            top, cnt = reasons.most_common(1)[0]   # most common blocker among time-feasible flights
            out[iid] = f"{top} on {cnt} of {feasible_in_time} time-feasible flights"
    return out
