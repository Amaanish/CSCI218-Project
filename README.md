# Gaza Humanitarian Air Bridge — Cargo Scheduling CSP

**CSCI218 — Artificial Intelligence**

A constraint satisfaction system that schedules humanitarian cargo (medical supplies, food, water, shelter) onto aircraft flight slots while respecting capacity, urgency, cold-chain, hazmat, and time-window constraints.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Team Contributions](#team-contributions)

---

## Project Overview

**Topic:** Humanitarian Cargo Scheduling as a Constraint Satisfaction Problem (CSP)

**Problem:** Schedule medical supplies, aircraft, cargo loads, and delivery windows while satisfying capacity, urgency, handling, customs, and airport-slot constraints.

**Approach:** Constraint satisfaction with backtracking search, heuristics (MRV, Degree, LCV), and arc consistency (AC-3). Performance is benchmarked against a greedy baseline.

**Output:** A feasible, prioritised cargo plan that maximises delivery of life-saving (P1) cargo and reports unmet demand and constraint conflicts.

### CSP Formulation

| Element | Description |
|---|---|
| **Variables** | One per cargo item X_i — which flight slot it is assigned to (or left behind) |
| **Domains** | All time-feasible, capacity-compatible flight slots + UNASSIGNED |
| **Hard constraints** | Time windows (C1), cold-chain support (C2), weight capacity (C3), volume capacity (C4), cold storage capacity (C5), hazmat segregation (C6) |
| **Soft constraints** | Minimise unmet demand weighted by priority; minimise delivery delay |

### Priority Tiers

| Tier | Description | Examples | Penalty weight |
|---|---|---|---|
| P1 | Life-saving | Insulin, blood, trauma kits | 100× |
| P2 | Urgent | Antibiotics, nutrition | 10× |
| P3 | Standard | Shelter, hygiene | 1× |

---

## Repository Structure

```
CSCI218-Project/
│
├── humanitarian_csp/          # Core CSP package
│   ├── model.py               # Data structures: CargoItem, Aircraft, FlightSlot, Instance
│   ├── csp.py                 # CSP formulation: variables, domains, constraints
│   ├── solver.py              # Baseline backtracking solver (branch-and-bound)
│   ├── optimization.py        # Heuristics: MRV, Degree, LCV, Forward Checking, AC-3
│   ├── evaluate.py            # Independent plan checker and metrics
│   ├── demo_instance.py       # Synthetic instance generator
│   └── __init__.py
│
├── experiments/
│   ├── greedy_baseline.py     # Greedy-Naive and Greedy-Feasible baselines
│   ├── run_experiments.py     # Repeated trials across small/medium/large datasets
│   ├── plot_results.py        # Generates charts from raw_results.csv
│   └── results/
│       ├── raw_results.csv    # Per-trial results for all algorithms
│       ├── summary.csv        # Mean/std summary table
│       └── fig1_time.png … fig7_cost.png
│
├── tests/
│   └── test_csp.py            # Unit tests
│
├── dashboard.py               # Streamlit visual dashboard
├── benchmark_optimization.py  # Heuristic configuration benchmarks
├── run_demo.py                 # Quick CLI demo
└── README.md
```

---

## Setup & Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/Amaanish/CSCI218-Project.git
cd CSCI218-Project

# Install dependencies
pip install streamlit matplotlib pandas numpy
```

No additional data files are needed — the synthetic instance generator is built in.

---

## How to Run

### 1. Quick CLI demo (25 items, default settings)

```bash
python run_demo.py
```

Optional arguments: `python run_demo.py [n_items] [seed] [capacity_factor] [time_limit_s]`

```bash
python run_demo.py 50 1 0.8 10
```

### 2. Visual dashboard (recommended)

```bash
streamlit run dashboard.py
```

Opens in your browser. Use the sidebar to configure the number of cargo requests, solver algorithm, capacity ratio, and time limit. The dashboard shows:
- Key metrics panel (violations, unmet demand, priority satisfaction)
- Gantt-style flight schedule coloured by priority tier
- Full cargo assignment table with constraint conflict log
- Experiment results charts (if `experiments/results/raw_results.csv` exists)

### 3. Run experiments (all algorithms, 3 dataset sizes)

```bash
python experiments/run_experiments.py
```

Options:
```bash
python experiments/run_experiments.py --trials 30 --time-limit 10 --capacity 0.8
```

Then generate charts:
```bash
python experiments/plot_results.py
```

### 4. Benchmark heuristic configurations

```bash
python benchmark_optimization.py
```

Outputs `benchmark_results.csv` comparing Baseline, MRV, Degree, MRV+LCV, MRV+LCV+FC, and MRV+LCV+FC+AC3 across dataset sizes 50–500.

### 5. Run unit tests

```bash
python -m pytest tests/
```

---
