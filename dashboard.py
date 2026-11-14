"""
dashboard.py -- Visual dashboard for the Gaza Humanitarian Air Bridge CSP.

Displays:
  - Live solver run with configurable parameters
  - Flight schedule (Gantt-style timeline)
  - Cargo assignment table
  - Key metrics panel
  - Experiment results (from experiments/results/raw_results.csv if present)

Usage:
    streamlit run dashboard.py
"""

import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# Make the package importable when launched from the project root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from humanitarian_csp import (
    HumanitarianCSP, BacktrackingSolver, evaluate, diagnose_unassigned, make_demo_instance,
)
from humanitarian_csp.optimization import (
    mrv_degree_selector, lcv_value_orderer, forward_checking, forward_checking_ac3,
)
from humanitarian_csp.solver import static_var_selector, static_value_orderer, no_inference

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gaza Humanitarian Air Bridge – CSP Dashboard",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Gaza Humanitarian Air Bridge — Cargo Scheduling CSP")
st.caption("CSCI218 · Group 5 · University of Wollongong in Dubai · Autumn 2026")

# ---------------------------------------------------------------------------
# Sidebar: solver parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Solver Parameters")

n_items = st.sidebar.slider("Number of cargo requests", 10, 200, 50, step=10)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999, value=1)
capacity_factor = st.sidebar.slider(
    "Airlift capacity / demand ratio",
    0.3, 1.5, 0.8, step=0.05,
    help="Values below 1.0 mean total airlift is less than total demand, forcing trade-offs.",
)
time_limit = st.sidebar.slider("Solver time limit (s)", 1, 30, 10)

algo_label = st.sidebar.selectbox(
    "Algorithm",
    ["CSP – Baseline Backtracking",
     "CSP + MRV + LCV + Forward Checking",
     "CSP + MRV + LCV + FC + AC-3"],
)

ALGO_MAP = {
    "CSP – Baseline Backtracking":
        (static_var_selector, static_value_orderer, no_inference),
    "CSP + MRV + LCV + Forward Checking":
        (mrv_degree_selector, lcv_value_orderer, forward_checking),
    "CSP + MRV + LCV + FC + AC-3":
        (mrv_degree_selector, lcv_value_orderer, forward_checking_ac3),
}

run_btn = st.sidebar.button("▶  Run Solver", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helper: Gantt chart
# ---------------------------------------------------------------------------
PRIORITY_COLOURS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c"}
PRIORITY_LABELS  = {1: "P1 – Life-saving", 2: "P2 – Urgent", 3: "P3 – Standard"}


def gantt_figure(instance, plan):
    """Return a Matplotlib figure showing each flight slot as a horizontal bar,
    coloured by the mix of priorities loaded."""
    slots  = {s.slot_id: s for s in instance.slots}
    items  = {it.item_id: it for it in instance.items}
    ac     = instance.aircraft_by_id()

    on_slot = defaultdict(list)
    for iid, sid in plan.items():
        if sid and sid in slots:
            on_slot[sid].append(items[iid])

    slot_list = sorted(instance.slots, key=lambda s: s.departure_h)
    fig, ax = plt.subplots(figsize=(11, max(3, len(slot_list) * 0.55 + 1.2)))

    for y, sl in enumerate(slot_list):
        plane  = ac[sl.aircraft_id]
        cargo  = on_slot[sl.slot_id]
        kg     = sum(c.weight_kg for c in cargo)
        pct    = kg / plane.payload_kg if plane.payload_kg else 0

        # background (empty capacity)
        ax.barh(y, sl.arrival_h - sl.departure_h, left=sl.departure_h,
                height=0.6, color="#e0e0e0", linewidth=0)

        # stacked priority colours proportional to weight
        left = sl.departure_h
        for pri in [1, 2, 3]:
            kg_p = sum(c.weight_kg for c in cargo if c.priority == pri)
            if kg_p > 0 and plane.payload_kg > 0:
                width = (kg_p / plane.payload_kg) * (sl.arrival_h - sl.departure_h)
                ax.barh(y, width, left=left, height=0.6,
                        color=PRIORITY_COLOURS[pri], linewidth=0, alpha=0.85)
                left += width

        label = (f"{sl.slot_id} · {plane.model}  "
                 f"{len(cargo)} items · {kg:,.0f}/{plane.payload_kg:,.0f} kg ({pct*100:.0f}%)")
        ax.text(sl.departure_h + 0.3, y, label, va="center", fontsize=7.5)

    ax.set_yticks(range(len(slot_list)))
    ax.set_yticklabels([s.slot_id for s in slot_list], fontsize=8)
    ax.set_xlabel("Time (hours from t = 0)")
    ax.set_title("Flight Schedule — cargo load by priority tier")
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=PRIORITY_COLOURS[p], label=PRIORITY_LABELS[p])
               for p in [1, 2, 3]]
    patches.append(mpatches.Patch(color="#e0e0e0", label="Empty capacity"))
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: metrics cards
# ---------------------------------------------------------------------------
def metrics_panel(m, stats, elapsed_ms):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hard violations",   m["hard_violations_total"],
              delta="✓ zero" if m["hard_violations_total"] == 0 else None,
              delta_color="normal")
    c2.metric("Unmet demand",      f"{m['unmet_demand_pct_items']:.1f}% of items")
    ps = m["priority_satisfaction"]
    c3.metric("P1 served",  f"{ps.get('P1', 0) or 0:.1f}%")
    c4.metric("P2 served",  f"{ps.get('P2', 0) or 0:.1f}%")
    c5.metric("P3 served",  f"{ps.get('P3', 0) or 0:.1f}%")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Solver time",        f"{elapsed_ms:.0f} ms")
    c7.metric("Nodes explored",     f"{stats.nodes:,}"     if stats else "—")
    c8.metric("Backtracks",         f"{stats.backtracks:,}" if stats else "—")
    c9.metric("Constraint checks",  f"{stats.constraint_checks:,}" if stats else "—")
    c10.metric("Flights used",      f"{m['flights_used']} / {m['flights_available']}")


# ---------------------------------------------------------------------------
# Helper: cargo table
# ---------------------------------------------------------------------------
def cargo_table(instance, plan):
    rows = []
    items = {it.item_id: it for it in instance.items}
    slots = {s.slot_id: s for s in instance.slots}
    for iid, sid in sorted(plan.items()):
        it = items[iid]
        if sid and sid in slots:
            sl = slots[sid]
            status = f"✅ {sid}  dep {sl.departure_h:.1f}h → arr {sl.arrival_h:.1f}h"
        else:
            status = "❌ Not shipped"
        rows.append({
            "Item": iid,
            "Priority": f"P{it.priority}",
            "Weight (kg)": f"{it.weight_kg:,.1f}",
            "Cold-chain": "❄️" if it.cold_chain else "",
            "Hazmat": "⚠️" if it.hazmat else "",
            "Category": it.category,
            "Ready (h)": it.ready_h,
            "Deadline (h)": it.deadline_h,
            "Assignment": status,
        })
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Main: run solver when button pressed
# ---------------------------------------------------------------------------
if run_btn:
    sel, ord_, inf = ALGO_MAP[algo_label]

    with st.spinner("Building instance and running solver…"):
        instance = make_demo_instance(n_items=n_items, seed=int(seed),
                                      capacity_factor=capacity_factor)
        csp = HumanitarianCSP(instance)
        t0 = time.perf_counter()
        result = BacktrackingSolver(
            csp,
            var_selector=sel,
            value_orderer=ord_,
            inference=inf,
            time_limit_s=time_limit,
        ).solve()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    m = evaluate(instance, result.plan)

    st.subheader("📊 Key Metrics")
    metrics_panel(m, result.stats, elapsed_ms)

    if result.stats.timed_out:
        st.warning("⏱ Solver hit the time limit — solution may not be optimal.")
    elif result.proven_optimal:
        st.success("✓ Proven optimal solution found.")

    st.subheader("🗓 Flight Schedule")
    st.pyplot(gantt_figure(instance, result.plan))

    st.subheader("📦 Cargo Assignment")
    df_cargo = cargo_table(instance, result.plan)
    st.dataframe(df_cargo, use_container_width=True, height=400)

    # unmet demand diagnosis
    why = diagnose_unassigned(instance, result.plan)
    if why:
        with st.expander(f"⚠️ Unshipped items ({len(why)}) — constraint conflict log"):
            rows_why = [{"Item": k, "Reason": v} for k, v in why.items()]
            st.dataframe(pd.DataFrame(rows_why), use_container_width=True)

    # hard violation breakdown
    if m["hard_violations_total"] > 0:
        with st.expander("🚨 Hard-constraint violation breakdown"):
            st.json(m["hard_violations"])

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Solver** to generate a schedule.")

# ---------------------------------------------------------------------------
# Experiment results tab (always shown if CSV exists)
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(HERE, "experiments", "results", "raw_results.csv")
FIG_DIR  = os.path.join(HERE, "experiments", "results")

if os.path.exists(CSV_PATH):
    st.divider()
    st.subheader("📈 Experiment Results")
    df_raw = pd.read_csv(CSV_PATH)

    tab1, tab2, tab3 = st.tabs(["Summary Table", "Charts", "Raw Data"])

    with tab1:
        SIZE_ORDER = ["small", "medium", "large"]
        cols = ["time_ms", "total_cost", "hard_violations",
                "unmet_items_pct", "p1_served_pct", "p2_served_pct", "p3_served_pct"]
        df_raw["timed_out"] = df_raw["timed_out"].astype(float)
        summary = (df_raw.groupby(["size", "algorithm"])[cols]
                   .agg(["mean", "std"])
                   .round(2)
                   .reindex(SIZE_ORDER, level=0))
        summary.columns = [" ".join(c) for c in summary.columns]
        st.dataframe(summary, use_container_width=True)

    with tab2:
        FIGS = {
            "fig1_time.png":       "Execution Time (ms, log scale)",
            "fig2_violations.png": "Hard-Constraint Violations",
            "fig3_unmet.png":      "Unmet Demand (% of items)",
            "fig4_p1.png":         "P1 Priority Satisfaction (%)",
            "fig5_p2.png":         "P2 Priority Satisfaction (%)",
            "fig6_p3.png":         "P3 Priority Satisfaction (%)",
            "fig7_cost.png":       "Total Objective Cost",
        }
        cols_figs = st.columns(2)
        for idx, (fname, caption) in enumerate(FIGS.items()):
            fpath = os.path.join(FIG_DIR, fname)
            if os.path.exists(fpath):
                cols_figs[idx % 2].image(fpath, caption=caption, use_container_width=True)

    with tab3:
        st.dataframe(df_raw, use_container_width=True)
