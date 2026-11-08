"""
model.py -- Plain data structures for the Humanitarian Cargo Scheduling problem.

These classes describe the *problem instance* (the raw input).  They know
nothing about search.  The CSP formulation built on top of them lives in csp.py.

All times are in hours from t = 0 (the start of the planning horizon).
All masses are in kg and all volumes in m^3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List

# Default priority weights: how "expensive" it is to fail a tier / delay it.
# P1 = life-saving (e.g. insulin, blood, trauma kits)
# P2 = urgent       (e.g. antibiotics, nutrition)
# P3 = standard     (e.g. shelter, hygiene)
DEFAULT_PRIORITY_WEIGHTS: Dict[int, float] = {1: 100.0, 2: 10.0, 3: 1.0}


@dataclass(frozen=True)
class CargoItem:
    """One cargo request (a consolidated pallet / shipment)."""
    item_id: str
    weight_kg: float
    volume_m3: float
    priority: int                 # 1, 2 or 3  (P1 most urgent)
    cold_chain: bool = False      # needs temperature-controlled storage
    hazmat: bool = False          # e.g. oxygen cylinders -> segregation rules
    category: str = "medical"     # medical | food | water | shelter | hygiene ...
    ready_h: float = 0.0          # earliest time it is cleared by customs & available at the hub
    deadline_h: float = 72.0      # latest acceptable arrival time at the destination
    handling_h: float = 2.0       # ground handling / loading time needed before departure


@dataclass(frozen=True)
class Aircraft:
    """A cargo aircraft (tail) with its physical limits."""
    aircraft_id: str
    model: str
    payload_kg: float             # max cargo mass
    volume_m3: float              # max cargo volume
    cold_volume_m3: float = 0.0   # temperature-controlled volume (0 = no cold storage)
    turnaround_h: float = 10.0    # min time between two departures of this aircraft


@dataclass(frozen=True)
class FlightSlot:
    """
    One scheduled sortie: a specific aircraft flying into a specific
    airport landing slot.  Cargo is assigned to flight slots.
    """
    slot_id: str
    aircraft_id: str
    departure_h: float
    arrival_h: float              # landing time = the delivery time of everything on board


@dataclass
class Instance:
    """A complete problem instance (this is what the JSON loader must produce)."""
    items: List[CargoItem]
    aircraft: List[Aircraft]
    slots: List[FlightSlot]
    priority_weights: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_PRIORITY_WEIGHTS))
    delay_rate: float = 0.2       # worst-case delay penalty as a fraction of the "unmet" penalty
    name: str = "instance"

    # ---- lookups ---------------------------------------------------------
    def aircraft_by_id(self) -> Dict[str, Aircraft]:
        return {a.aircraft_id: a for a in self.aircraft}

    @property
    def horizon_h(self) -> float:
        """Length of the planning horizon, used to normalise delay costs."""
        return max([it.deadline_h for it in self.items] + [1.0])

    # ---- objective (soft constraints) -----------------------------------
    def unmet_cost(self, item: CargoItem) -> float:
        """Penalty for leaving `item` undelivered: priority weight x tonnage."""
        return self.priority_weights[item.priority] * item.weight_kg

    def delay_cost(self, item: CargoItem, slot: FlightSlot) -> float:
        """Penalty for delivering `item` late: a fraction of unmet_cost, growing with waiting time."""
        wait = max(0.0, slot.arrival_h - item.ready_h)
        return self.delay_rate * self.unmet_cost(item) * wait / self.horizon_h

    # ---- (de)serialisation, shared with the data-pipeline teammate -------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "priority_weights": {str(k): v for k, v in self.priority_weights.items()},
            "delay_rate": self.delay_rate,
            "items": [asdict(i) for i in self.items],
            "aircraft": [asdict(a) for a in self.aircraft],
            "slots": [asdict(s) for s in self.slots],
        }

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        pw = {int(k): float(v) for k, v in d.get("priority_weights", DEFAULT_PRIORITY_WEIGHTS).items()}
        return Instance(
            items=[CargoItem(**i) for i in d["items"]],
            aircraft=[Aircraft(**a) for a in d["aircraft"]],
            slots=[FlightSlot(**s) for s in d["slots"]],
            priority_weights=pw,
            delay_rate=d.get("delay_rate", 0.2),
            name=d.get("name", "instance"),
        )

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_json(path: str) -> "Instance":
        with open(path) as f:
            return Instance.from_dict(json.load(f))

    # ---- sanity check on the instance itself -----------------------------
    def validate(self) -> List[str]:
        """Return a list of problems with the input data (empty list = OK)."""
        problems: List[str] = []
        ac = self.aircraft_by_id()
        for s in self.slots:
            if s.aircraft_id not in ac:
                problems.append(f"slot {s.slot_id}: unknown aircraft {s.aircraft_id}")
            if s.arrival_h < s.departure_h:
                problems.append(f"slot {s.slot_id}: arrives before it departs")
        # an aircraft cannot fly two sorties closer together than its turnaround time
        by_ac: Dict[str, List[FlightSlot]] = {}
        for s in self.slots:
            by_ac.setdefault(s.aircraft_id, []).append(s)
        for aid, lst in by_ac.items():
            lst.sort(key=lambda s: s.departure_h)
            for a, b in zip(lst, lst[1:]):
                if aid in ac and b.departure_h - a.departure_h < ac[aid].turnaround_h:
                    problems.append(f"aircraft {aid}: {a.slot_id}->{b.slot_id} violates turnaround time")
        for it in self.items:
            if it.priority not in self.priority_weights:
                problems.append(f"item {it.item_id}: priority {it.priority} has no weight")
        ids = [i.item_id for i in self.items]
        if len(ids) != len(set(ids)):
            problems.append("duplicate item_id values")
        return problems
