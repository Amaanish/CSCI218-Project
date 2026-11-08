"""
csp.py -- The CSP (constraint optimisation) formulation of the problem.

FORMAL MODEL
------------
Variables   X_i, one per cargo item i = 0..n-1.
Domains     D_i = { s : slot s is time-feasible and big enough for item i }  U  { UNASSIGNED }
            ("UNASSIGNED" = -1 means the item is left behind / unmet demand.)

Hard constraints (must never be violated)
    C1  Time window (unary):  ready_i + handling_i <= departure(s)   and   arrival(s) <= deadline_i
    C2  Cold-chain support (unary): a cold-chain item may only ride on an aircraft that has
        enough cold storage for it.
    C3  Weight capacity (n-ary, one per slot):   sum_{i: X_i = s} weight_i <= payload(s)
    C4  Volume capacity (n-ary, one per slot):   sum_{i: X_i = s} volume_i <= volume(s)
    C5  Cold storage capacity (n-ary, one per slot):
                                                 sum_{i: X_i = s, cold_i} volume_i <= cold_volume(s)
    C6  Hazmat segregation (binary): for items i != j where one is hazmat and the other is
        "sensitive" (cold-chain or food):  X_i = X_j = s  is forbidden for every real slot s.

Soft constraints (encoded as the objective to MINIMISE)
    S1  Unmet demand:  leaving item i behind costs  W[priority_i] * weight_i
    S2  Timeliness:    delivering item i costs      delay_rate * S1_cost_i * (arrival - ready_i) / horizon

C1 and C2 are *unary*: they are applied once, up front, by "node consistency" (build_domains).
C3-C5 are *global* capacity constraints: they depend on everything already placed in a slot,
so they are checked incrementally through the Assignment class.
C6 is a genuine *binary* constraint: it is the one AC-3 can work on directly.

Because the soft constraints give us a cost to minimise, formally this is a Constraint
Optimisation Problem (COP).  The solver in solver.py is a depth-first backtracking search
with branch-and-bound on that cost.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from .model import Instance

UNASSIGNED = -1          # domain value meaning "this item is not shipped"
_EPS = 1e-9


class HumanitarianCSP:
    """Static description of the problem: variables, domains, constraint data, costs."""

    def __init__(self, instance: Instance):
        self.instance = instance
        self.items = instance.items
        self.slots = instance.slots
        self.n_vars = len(self.items)
        self.n_slots = len(self.slots)

        ac = instance.aircraft_by_id()
        self.slot_aircraft = [ac[s.aircraft_id] for s in self.slots]
        self.cap_weight = [a.payload_kg for a in self.slot_aircraft]
        self.cap_volume = [a.volume_m3 for a in self.slot_aircraft]
        self.cap_cold = [a.cold_volume_m3 for a in self.slot_aircraft]

        # "sensitive" cargo must not share a flight with hazmat (constraint C6)
        self.sensitive = [it.cold_chain or it.category == "food" for it in self.items]
        self.hazmat = [it.hazmat for it in self.items]
        self.conflict_adj: List[Set[int]] = self._build_conflict_graph()

        # objective data
        self.unmet_cost = [instance.unmet_cost(it) for it in self.items]
        self.assigned_cost = [
            [instance.delay_cost(it, sl) for sl in self.slots] for it in self.items
        ]

        self.initial_domains: Dict[int, List[int]] = self.build_domains()

    # ------------------------------------------------------------------ domains
    def unary_ok(self, i: int, s: int) -> bool:
        """Constraints C1 + C2 (+ 'item fits in an empty aircraft'): can item i EVER go on slot s?"""
        it, sl = self.items[i], self.slots[s]
        if it.ready_h + it.handling_h > sl.departure_h + _EPS:        # C1: not loaded in time
            return False
        if sl.arrival_h > it.deadline_h + _EPS:                       # C1: arrives too late
            return False
        if it.weight_kg > self.cap_weight[s] + _EPS:                  # alone it is too heavy
            return False
        if it.volume_m3 > self.cap_volume[s] + _EPS:                  # alone it is too bulky
            return False
        if it.cold_chain and it.volume_m3 > self.cap_cold[s] + _EPS:  # C2: no cold storage
            return False
        return True

    def build_domains(self) -> Dict[int, List[int]]:
        """Node consistency: keep only slots that satisfy the unary constraints.
        Slots are listed in input order; UNASSIGNED is always last so that the
        default search tries to ship an item before giving up on it."""
        return {
            i: [s for s in range(self.n_slots) if self.unary_ok(i, s)] + [UNASSIGNED]
            for i in range(self.n_vars)
        }

    def fresh_domains(self) -> Dict[int, List[int]]:
        """A mutable copy of the initial domains (search algorithms prune these)."""
        return {i: list(d) for i, d in self.initial_domains.items()}

    # ------------------------------------------------------------ binary constraint
    def _build_conflict_graph(self) -> List[Set[int]]:
        adj: List[Set[int]] = [set() for _ in range(self.n_vars)]
        haz = [i for i in range(self.n_vars) if self.hazmat[i]]
        sens = [j for j in range(self.n_vars) if self.sensitive[j]]
        for i in haz:
            for j in sens:
                if i != j:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def binary_ok(self, i: int, vi: int, j: int, vj: int) -> bool:
        """Constraint C6 as a function of two (variable, value) pairs (what AC-3 needs)."""
        if vi == UNASSIGNED or vj == UNASSIGNED:
            return True
        if j in self.conflict_adj[i]:
            return vi != vj
        return True

    def neighbors(self, i: int) -> Set[int]:
        """Variables that share an explicit binary constraint with i (for AC-3 / degree heuristic)."""
        return self.conflict_adj[i]

    # ------------------------------------------------------------------ misc
    def new_assignment(self) -> "Assignment":
        return Assignment(self)

    def plan_from_values(self, values: List[Optional[int]]) -> Dict[str, Optional[str]]:
        """Convert internal integers into {item_id: slot_id or None}."""
        return {
            self.items[i].item_id: (None if v in (None, UNASSIGNED) else self.slots[v].slot_id)
            for i, v in enumerate(values)
        }


class Assignment:
    """
    A (partial) assignment, with the per-slot load totals kept up to date incrementally
    so that checking a new placement is O(1) instead of re-summing every item.

    value[i] is None while item i is undecided, UNASSIGNED (-1) if left behind,
    or a slot index.
    """

    def __init__(self, csp: HumanitarianCSP):
        self.csp = csp
        n, S = csp.n_vars, csp.n_slots
        self.value: List[Optional[int]] = [None] * n
        self.load_w = [0.0] * S
        self.load_v = [0.0] * S
        self.load_cold = [0.0] * S
        self.n_hazmat = [0] * S       # number of hazmat items on each slot
        self.n_sensitive = [0] * S    # number of cold-chain/food items on each slot
        self.cost = 0.0
        self.n_decided = 0
        self.constraint_checks = 0    # counted for the benchmarking teammate

    # ---- constraint checking --------------------------------------------
    def violation(self, i: int, s: int) -> Optional[str]:
        """
        Would placing item i on slot s break a hard constraint, given what is already placed?
        Returns None if fine, otherwise a short reason code (useful for unmet-demand logs):
        'weight_full', 'volume_full', 'cold_storage_full', 'hazmat_conflict'.
        """
        self.constraint_checks += 1
        if s == UNASSIGNED:
            return None
        c, it = self.csp, self.csp.items[i]
        if self.load_w[s] + it.weight_kg > c.cap_weight[s] + _EPS:
            return "weight_full"
        if self.load_v[s] + it.volume_m3 > c.cap_volume[s] + _EPS:
            return "volume_full"
        if it.cold_chain and self.load_cold[s] + it.volume_m3 > c.cap_cold[s] + _EPS:
            return "cold_storage_full"
        if (c.hazmat[i] and self.n_sensitive[s] > 0) or (c.sensitive[i] and self.n_hazmat[s] > 0):
            return "hazmat_conflict"
        return None

    # ---- assign / undo ---------------------------------------------------
    def assign(self, i: int, s: int) -> None:
        c, it = self.csp, self.csp.items[i]
        self.value[i] = s
        self.n_decided += 1
        if s == UNASSIGNED:
            self.cost += c.unmet_cost[i]
            return
        self.load_w[s] += it.weight_kg
        self.load_v[s] += it.volume_m3
        if it.cold_chain:
            self.load_cold[s] += it.volume_m3
        if c.hazmat[i]:
            self.n_hazmat[s] += 1
        if c.sensitive[i]:
            self.n_sensitive[s] += 1
        self.cost += c.assigned_cost[i][s]

    def unassign(self, i: int) -> None:
        c, it = self.csp, self.csp.items[i]
        s = self.value[i]
        self.value[i] = None
        self.n_decided -= 1
        if s == UNASSIGNED:
            self.cost -= c.unmet_cost[i]
            return
        self.load_w[s] -= it.weight_kg
        self.load_v[s] -= it.volume_m3
        if it.cold_chain:
            self.load_cold[s] -= it.volume_m3
        if c.hazmat[i]:
            self.n_hazmat[s] -= 1
        if c.sensitive[i]:
            self.n_sensitive[s] -= 1
        self.cost -= c.assigned_cost[i][s]

    # ---- helpers for the heuristics teammate ----------------------------
    def residual(self, s: int):
        """Remaining (weight, volume, cold volume) on slot s."""
        c = self.csp
        return (c.cap_weight[s] - self.load_w[s],
                c.cap_volume[s] - self.load_v[s],
                c.cap_cold[s] - self.load_cold[s])

    def is_complete(self) -> bool:
        return self.n_decided == self.csp.n_vars
