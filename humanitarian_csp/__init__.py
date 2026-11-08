"""Humanitarian cargo scheduling as a CSP -- core model and baseline solver."""
from .model import Aircraft, CargoItem, FlightSlot, Instance
from .csp import HumanitarianCSP, Assignment, UNASSIGNED
from .solver import BacktrackingSolver, SolveResult, SearchStats
from .evaluate import evaluate, diagnose_unassigned
from .demo_instance import make_demo_instance
