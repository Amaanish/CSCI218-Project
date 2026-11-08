"""
solver.py -- Baseline Backtracking Search (depth-first, with branch-and-bound).

BASELINE BEHAVIOUR (no heuristics)
    * Variable order : the order the items appear in the input (static).
    * Value order    : slots in input order, then UNASSIGNED last.
    * Inference      : none (constraints are only checked when a value is tried).
    * Pruning        : a branch is abandoned as soon as its cost so far is no better
                       than the best complete plan found (branch-and-bound).

The three behaviours above are *hooks*, so the optimisation teammate can plug in
MRV / Degree / LCV and Forward Checking / AC-3 without touching the search loop:

    var_selector(csp, assignment, domains, unassigned) -> variable index
    value_orderer(csp, var, assignment, domains)       -> list of values (ordered)
    inference(csp, var, value, assignment, domains)    -> list of (var, value) pairs
                                                          that were pruned from `domains`,
                                                          or None if a domain was wiped out
                                                          (= dead end).  The solver restores
                                                          pruned values on backtrack.

Search is an "anytime" algorithm: UNASSIGNED is always in every domain, so a complete
(possibly poor) plan is found immediately, and the search then keeps improving it until it
finishes (optimal) or runs out of time / nodes.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .csp import HumanitarianCSP, UNASSIGNED


# ---------------------------------------------------------------- default hooks
def static_var_selector(csp, assignment, domains, unassigned):
    """Baseline: take the next variable in input order."""
    return unassigned[0]


def static_value_orderer(csp, var, assignment, domains):
    """Baseline: values in the order they appear in the domain."""
    return list(domains[var])


def no_inference(csp, var, value, assignment, domains):
    """Baseline: no propagation."""
    return []


# ---------------------------------------------------------------- result objects
@dataclass
class SearchStats:
    nodes: int = 0                 # (variable, value) pairs tried
    backtracks: int = 0            # times a variable ran out of values
    constraint_checks: int = 0     # calls to Assignment.violation
    rejected: Dict[str, int] = field(default_factory=dict)   # violation reason -> count
    solutions_found: int = 0       # improving complete plans found
    elapsed_s: float = 0.0
    timed_out: bool = False        # stopped by time/node budget (so optimality not proven)


@dataclass
class SolveResult:
    plan: Dict[str, Optional[str]]      # item_id -> slot_id, or None if not shipped
    cost: float
    proven_optimal: bool                # True only if the search space was exhausted
    stats: SearchStats


class BacktrackingSolver:
    def __init__(
        self,
        csp: HumanitarianCSP,
        var_selector: Callable = static_var_selector,
        value_orderer: Callable = static_value_orderer,
        inference: Callable = no_inference,
        time_limit_s: Optional[float] = 10.0,
        node_limit: Optional[int] = None,
        first_solution_only: bool = False,
    ):
        self.csp = csp
        self.var_selector = var_selector
        self.value_orderer = value_orderer
        self.inference = inference
        self.time_limit_s = time_limit_s
        self.node_limit = node_limit
        self.first_solution_only = first_solution_only

    # ------------------------------------------------------------------ public
    def solve(self) -> SolveResult:
        csp = self.csp
        self.assignment = csp.new_assignment()
        self.domains = csp.fresh_domains()
        self.unassigned: List[int] = list(range(csp.n_vars))
        self.stats = SearchStats()
        self.best_cost = float("inf")
        self.best_values: List[Optional[int]] = [UNASSIGNED] * csp.n_vars
        self._stop = False
        self._t0 = time.perf_counter()

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, csp.n_vars * 3 + 1000))   # depth = number of items
        try:
            self._search()
        finally:
            sys.setrecursionlimit(old_limit)

        self.stats.elapsed_s = time.perf_counter() - self._t0
        self.stats.constraint_checks = self.assignment.constraint_checks
        proven = not self.stats.timed_out and not (self.first_solution_only and self._stop)
        return SolveResult(csp.plan_from_values(self.best_values), self.best_cost, proven, self.stats)

    # ------------------------------------------------------------------ search
    def _out_of_budget(self) -> bool:
        st = self.stats
        if self.node_limit is not None and st.nodes >= self.node_limit:
            st.timed_out = True
        elif (self.time_limit_s is not None and st.nodes % 256 == 0
              and time.perf_counter() - self._t0 > self.time_limit_s):
            st.timed_out = True
        return st.timed_out

    def _search(self) -> None:
        a, csp, st = self.assignment, self.csp, self.stats

        # ---- goal test: every variable has a value -> record the new best plan
        if not self.unassigned:
            if a.cost < self.best_cost - 1e-9:
                self.best_cost = a.cost
                self.best_values = list(a.value)
                st.solutions_found += 1
                if self.first_solution_only:
                    self._stop = True
            return

        # ---- 1. choose a variable (hook)
        var = self.var_selector(csp, a, self.domains, self.unassigned)
        pos = self.unassigned.index(var)
        self.unassigned.pop(pos)

        # ---- 2. try its values in order (hook)
        for val in self.value_orderer(csp, var, a, self.domains):
            if self._stop or self._out_of_budget():
                break
            st.nodes += 1

            # branch-and-bound: this branch can only get more expensive
            step = csp.unmet_cost[var] if val == UNASSIGNED else csp.assigned_cost[var][val]
            if a.cost + step >= self.best_cost - 1e-9:
                continue

            # hard-constraint check
            reason = a.violation(var, val)
            if reason is not None:
                st.rejected[reason] = st.rejected.get(reason, 0) + 1
                continue

            a.assign(var, val)
            pruned = self.inference(csp, var, val, a, self.domains)   # hook (None = dead end)
            if pruned is not None:
                self._search()
                for (v, x) in pruned:                                 # undo propagation
                    self.domains[v].append(x)
            a.unassign(var)

        # ---- 3. backtrack: put the variable back
        self.unassigned.insert(pos, var)
        st.backtracks += 1
