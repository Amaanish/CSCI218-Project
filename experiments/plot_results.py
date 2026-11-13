"""
plot_results.py -- Summary table (mean/std) + graphs from raw_results.csv.
Usage:  python experiments/plot_results.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
df = pd.read_csv(os.path.join(RES, "raw_results.csv"))

SIZE_ORDER = ["small", "medium", "large"]
ALGOS = list(df["algorithm"].unique())

# ---------- summary table: mean and std per (size, algorithm) ----------
cols = ["time_ms", "total_cost", "hard_violations", "unmet_items_pct", "unmet_weight_pct",
        "p1_served_pct", "p2_served_pct", "p3_served_pct", "timed_out"]
df["timed_out"] = df["timed_out"].astype(float)
summary = df.groupby(["size", "algorithm"])[cols].agg(["mean", "std"]).round(2)
summary = summary.reindex(SIZE_ORDER, level=0)
summary.to_csv(os.path.join(RES, "summary.csv"))
print(summary.to_string())


# ---------- grouped bar chart helper ----------
def bar_chart(metric, ylabel, title, fname, log=False):
    x = np.arange(len(SIZE_ORDER))
    width = 0.8 / len(ALGOS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, algo in enumerate(ALGOS):
        sub = df[df["algorithm"] == algo].groupby("size")[metric]
        means = [sub.mean().get(s, np.nan) for s in SIZE_ORDER]
        stds = [sub.std().get(s, 0) for s in SIZE_ORDER]
        ax.bar(x + k * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.capitalize()}\n({df[df['size']==s]['n_items'].iloc[0]} items)" for s in SIZE_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, fname), dpi=200)
    plt.close(fig)


bar_chart("time_ms", "Execution time (ms, log scale)", "Execution time by dataset size", "fig1_time.png", log=True)
bar_chart("hard_violations", "Hard-constraint violations (count)", "Constraint violations by dataset size", "fig2_violations.png")
bar_chart("unmet_items_pct", "Unmet demand (% of items)", "Unmet demand by dataset size", "fig3_unmet.png")
bar_chart("p1_served_pct", "P1 items delivered (%)", "Priority satisfaction: P1 (life-saving)", "fig4_p1.png")
bar_chart("p2_served_pct", "P2 items delivered (%)", "Priority satisfaction: P2 (urgent)", "fig5_p2.png")
bar_chart("p3_served_pct", "P3 items delivered (%)", "Priority satisfaction: P3 (standard)", "fig6_p3.png")
bar_chart("total_cost", "Objective cost (lower is better)", "Total solution cost by dataset size", "fig7_cost.png", log=True)
print("Saved figures and summary.csv to", RES)