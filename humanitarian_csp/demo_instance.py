"""
demo_instance.py -- A small, seeded instance generator used ONLY to test and demo the solver.

The data-pipeline teammate owns the real generator (50-500 requests, P1-P3 tiers, etc.).
It must produce the same `Instance` structure (see model.py / Instance.to_dict), so the
two parts plug together through the JSON format.
"""
from __future__ import annotations

import random
from typing import List

from .model import Aircraft, CargoItem, FlightSlot, Instance

FLIGHT_TIME_H = 4.0      # Dubai -> destination


def make_demo_instance(n_items: int = 30, seed: int = 0, capacity_factor: float = 0.8) -> Instance:
    """capacity_factor < 1 means total airlift capacity is below total demand (so some cargo must be left)."""
    rng = random.Random(seed)

    fleet = [
        Aircraft("AC1", "Heavy freighter", 40000, 200, cold_volume_m3=20, turnaround_h=10),
        Aircraft("AC2", "Heavy freighter", 40000, 200, cold_volume_m3=20, turnaround_h=10),
        Aircraft("AC3", "Medium freighter", 18000, 90, cold_volume_m3=10, turnaround_h=10),
        Aircraft("AC4", "Medium freighter", 18000, 90, cold_volume_m3=10, turnaround_h=10),
        Aircraft("AC5", "Light cargo", 6000, 30, cold_volume_m3=0, turnaround_h=8),
        Aircraft("AC6", "Light cargo", 6000, 30, cold_volume_m3=0, turnaround_h=8),
    ]

    # ---- cargo requests
    items: List[CargoItem] = []
    for k in range(n_items):
        priority = rng.choices([1, 2, 3], weights=[20, 35, 45])[0]
        cold = rng.random() < (0.30 if priority == 1 else 0.10)
        hazmat = (not cold) and rng.random() < 0.05
        category = "medical" if (cold or hazmat or priority == 1) else rng.choice(["food", "water", "shelter", "hygiene"])
        if cold:
            weight = rng.uniform(20, 300)
            volume = rng.uniform(0.5, 4.0)
        else:
            weight = rng.uniform(100, 1500)
            volume = weight / rng.uniform(80, 250)
        ready = rng.uniform(0, 36)
        window = {1: (24, 48), 2: (36, 72), 3: (48, 96)}[priority]
        items.append(CargoItem(
            item_id=f"P{priority}-{k:03d}", weight_kg=round(weight, 1), volume_m3=round(volume, 2),
            priority=priority, cold_chain=cold, hazmat=hazmat, category=category,
            ready_h=round(ready, 1), deadline_h=round(ready + rng.uniform(*window), 1),
            handling_h=2.0,
        ))

    # ---- flight slots: round-robin over the fleet until capacity ~ capacity_factor x demand,
    # then spread the departures evenly over the planning horizon.
    demand = sum(it.weight_kg for it in items)
    n_slots, capacity = 0, 0.0
    while capacity < capacity_factor * demand or n_slots < 6:
        capacity += fleet[n_slots % len(fleet)].payload_kg
        n_slots += 1
    step = max(2.0, 66.0 / max(1, n_slots - 1))      # hours between consecutive departures
    slots: List[FlightSlot] = []
    for k in range(n_slots):
        plane = fleet[k % len(fleet)]
        dep = round(6.0 + k * step, 1)
        slots.append(FlightSlot(f"F{k + 1:02d}", plane.aircraft_id, dep, round(dep + FLIGHT_TIME_H, 1)))

    return Instance(items=items, aircraft=fleet, slots=slots, name=f"demo_{n_items}_seed{seed}")
