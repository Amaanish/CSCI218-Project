"""
Optimization hooks for the Humanitarian Cargo Scheduling CSP.

Drop this file into the humanitarian_csp package as optimization.py
(or adjust the import in benchmark_optimization.py).

Implements:
  - MRV (Minimum Remaining Values)
  - Degree heuristic (tie-breaker for MRV, plus standalone selector)
  - LCV (Least Constraining Value)
  - Forward Checking
  - AC-3 over the CSP's explicit binary conflict graph

Important modelling note:
  Capacity constraints C3-C5 are global/n-ary, so Forward Checking handles
  them using the current residual aircraft capacities. AC-3 handles C6,
  the explicit hazmat-versus-sensitive binary constraint. Because UNASSIGNED
  remains a legal fallback value, AC-3 may make few or no extra removals;
  this is correct, not a bug.
"""
from collections import deque

from .csp import UNASSIGNED


# ---------------------------------------------------------------------------
# Variable-ordering heuristics

def mrv_selector(csp, assignment, domains, unassigned):
    """Choose the undecided item with the fewest remaining legal values."""
    return min(unassigned, key=lambda v: (len(domains[v]), v))


def degree_selector(csp, assignment, domains, unassigned):
    """Choose the item connected to the most other undecided items."""
    undecided = set(unassigned)
    return max(
        unassigned,
        key=lambda v: (sum(1 for n in csp.neighbors(v) if n in undecided), -v),
    )


def mrv_degree_selector(csp, assignment, domains, unassigned):
    """
    MRV first; among ties, choose the variable with the highest degree
    in the remaining constraint graph.
    """
    undecided = set(unassigned)

    def key(v):
        degree = sum(1 for n in csp.neighbors(v) if n in undecided)
        return (len(domains[v]), -degree, v)

    return min(unassigned, key=key)


# ---------------------------------------------------------------------------
# Value ordering

def _estimated_pruned_values(csp, var, value, assignment, domains):
    """
    Estimate how many choices this value removes from other undecided items.

    Counts:
      1. Values ruled out by binary hazmat/sensitive conflicts.
      2. Same-flight values that would no longer fit the residual weight,
         volume, or cold-storage capacity after this item is placed.

    This is an ordering estimate only; Forward Checking performs the
    authoritative pruning after the value is actually assigned.
    """
    if value == UNASSIGNED:
        return 0

    item = csp.items[var]
    rem_w, rem_v, rem_cold = assignment.residual(value)
    rem_w -= item.weight_kg
    rem_v -= item.volume_m3
    if item.cold_chain:
        rem_cold -= item.volume_m3

    score = 0
    for other in range(csp.n_vars):
        if other == var or assignment.value[other] is not None:
            continue

        other_item = csp.items[other]
        for other_value in domains[other]:
            if other_value == UNASSIGNED:
                continue

            # Explicit binary constraint.
            if not csp.binary_ok(var, value, other, other_value):
                score += 1
                continue

            # Global capacity impact if the other item uses this same slot.
            if other_value == value:
                if (rem_w + 1e-9 < other_item.weight_kg
                    or rem_v + 1e-9 < other_item.volume_m3
                    or (other_item.cold_chain
                        and rem_cold + 1e-9 < other_item.volume_m3)):
                    score += 1

    return score


def lcv_value_orderer(csp, var, assignment, domains):
    """
    Try the least-constraining values first.

    UNASSIGNED is deliberately kept last so the search first explores
    shipment options, matching the baseline solver's anytime behaviour.
    """
    values = list(domains[var])
    real_values = [v for v in values if v != UNASSIGNED]
    real_values.sort(
        key=lambda v: (
            _estimated_pruned_values(csp, var, v, assignment, domains),
            csp.assigned_cost[var][v],
        )
    )
    if UNASSIGNED in values:
        real_values.append(UNASSIGNED)
    return real_values


# ---------------------------------------------------------------------------
# Inference: Forward Checking + AC-3

def forward_checking(csp, var, value, assignment, domains):
    """
    Remove values from remaining variables' domains that are no longer
    feasible after the latest assignment.

    The solver expects:
      - list[(variable_index, removed_value)] on success
      - None if a domain becomes empty

    If a wipeout occurs, this function restores its own changes before
    returning None. This is necessary because the supplied solver only
    restores the returned prune-list when inference succeeds.
    """
    pruned = []

    if value == UNASSIGNED:
        return pruned

    for other in range(csp.n_vars):
        if assignment.value[other] is not None:
            continue

        for candidate in list(domains[other]):
            if candidate == UNASSIGNED:
                continue
            if assignment.violation(other, candidate) is not None:
                domains[other].remove(candidate)
                pruned.append((other, candidate))

        if not domains[other]:
            for v, x in reversed(pruned):
                domains[v].append(x)
            return None

    return pruned


def ac3(csp, domains, initial_arcs=None, pruned=None):
    """
    AC-3 for the explicit binary constraints in csp.conflict_adj.

    Domains include UNASSIGNED. Since leaving an item unshipped is always
    permitted, UNASSIGNED provides support for every value under the
    hazmat/sensitive binary constraint. Thus AC-3 is logically sound but
    can be weak on this particular model. Forward Checking is what removes
    values made infeasible by the current partial assignment and capacities.

    `pruned` is an existing list to append removals to, allowing the solver
    to restore all changes on backtracking.
    """
    if pruned is None:
        pruned = []

    if initial_arcs is None:
        queue = deque(
            (i, j)
            for i in range(csp.n_vars)
            for j in csp.neighbors(i)
            if domains[i] and domains[j]
        )
    else:
        queue = deque(initial_arcs)

    while queue:
        xi, xj = queue.popleft()
        revised = False

        for x in list(domains[xi]):
            # x is supported if some y in Xj's domain satisfies the binary
            # constraint. UNASSIGNED is intentionally included as a value.
            if not any(
                csp.binary_ok(xi, x, xj, y)
                for y in domains[xj]
            ):
                domains[xi].remove(x)
                pruned.append((xi, x))
                revised = True

        if not domains[xi]:
            return None

        if revised:
            for xk in csp.neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))

    return pruned


def forward_checking_ac3(csp, var, value, assignment, domains):
    """
    Combined inference hook for BacktrackingSolver.

    First applies Forward Checking to current aircraft loads and conflicts,
    then runs AC-3 on the remaining binary-constraint graph.
    """
    pruned = forward_checking(csp, var, value, assignment, domains)
    if pruned is None:
        return None

    result = ac3(csp, domains, pruned=pruned)
    if result is None:
        # Restore all changes before telling the solver this branch failed.
        for v, x in reversed(pruned):
            domains[v].append(x)
        return None

    return pruned
