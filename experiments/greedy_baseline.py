"""
greedy_baseline.py -- Greedy baselines for comparison with the CSP solver.

Both sort cargo heaviest-first and put each item on the first flight (earliest
departure) that accepts it. They differ in which constraints they respect:

  greedy_naive    : only checks weight + volume capacity  ("pure weight loading")
  greedy_feasible : also respects time windows, cold-chain storage and hazmat rules

Both return a plan {item_id: slot_id or None}, which evaluate() can score.
"""
from __future__ import annotations


def greedy_naive(instance):
    ac = instance.aircraft_by_id()
    slots = sorted(instance.slots, key=lambda s: s.departure_h)
    load = {s.slot_id: [0.0, 0.0] for s in slots}          # [weight, volume]
    plan = {}
    for it in sorted(instance.items, key=lambda i: i.weight_kg, reverse=True):
        plan[it.item_id] = None
        for sl in slots:
            plane = ac[sl.aircraft_id]
            w, v = load[sl.slot_id]
            if w + it.weight_kg <= plane.payload_kg and v + it.volume_m3 <= plane.volume_m3:
                load[sl.slot_id] = [w + it.weight_kg, v + it.volume_m3]
                plan[it.item_id] = sl.slot_id
                break
    return plan


def greedy_feasible(instance):
    ac = instance.aircraft_by_id()
    slots = sorted(instance.slots, key=lambda s: s.departure_h)
    load = {s.slot_id: {"w": 0.0, "v": 0.0, "cold": 0.0, "haz": 0, "sens": 0} for s in slots}
    plan = {}
    for it in sorted(instance.items, key=lambda i: i.weight_kg, reverse=True):
        plan[it.item_id] = None
        sensitive = it.cold_chain or it.category == "food"
        for sl in slots:
            plane = ac[sl.aircraft_id]
            L = load[sl.slot_id]
            if it.ready_h + it.handling_h > sl.departure_h:   # not loaded in time
                continue
            if sl.arrival_h > it.deadline_h:                  # arrives too late
                continue
            if L["w"] + it.weight_kg > plane.payload_kg:
                continue
            if L["v"] + it.volume_m3 > plane.volume_m3:
                continue
            if it.cold_chain and L["cold"] + it.volume_m3 > plane.cold_volume_m3:
                continue
            if (it.hazmat and L["sens"] > 0) or (sensitive and L["haz"] > 0):
                continue
            L["w"] += it.weight_kg
            L["v"] += it.volume_m3
            if it.cold_chain:
                L["cold"] += it.volume_m3
            L["haz"] += int(it.hazmat)
            L["sens"] += int(sensitive)
            plan[it.item_id] = sl.slot_id
            break
    return plan