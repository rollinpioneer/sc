"""Render the four HB1 diagnostic figures from frozen metric tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recovery_handback.common import read_table


ORDER = ["none", "l5", "l20", "l80", "full"]
LABELS = {"none": "No help", "l5": "5", "l20": "20", "l80": "80", "full": "Full (no handoff)"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.metrics_dir / "curves_by_task.csv")
    opportunities = pd.DataFrame(read_table(args.metrics_dir / "handoff_opportunities.parquet"))
    curves.to_csv(args.output_dir / "plot_curves_by_task.csv", index=False)

    tasks = list(dict.fromkeys(curves["task"].dropna().astype(str)))
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for task in tasks:
        rows = curves[curves["task"] == task].set_index("branch_name").reindex(ORDER)
        values = rows["system_success_rate_root"].astype(float).to_numpy()
        lower = values - rows["system_success_ci95_low"].astype(float).to_numpy()
        upper = rows["system_success_ci95_high"].astype(float).to_numpy() - values
        ax.errorbar(x, values, yerr=np.vstack([lower, upper]), marker="o", capsize=3, label=task)
    ax.set_xticks(x, [LABELS[item] for item in ORDER])
    ax.set_ylabel("Root-clustered system success")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    _save(fig, args.output_dir / "01_system_success_by_help_length.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for task in tasks:
        rows = curves[(curves["task"] == task) & curves["branch_name"].isin(["l5", "l20", "l80"])].set_index("branch_name").reindex(["l5", "l20", "l80"])
        ax.plot([5, 20, 80], rows["rescue_fraction_among_baseline_failed_anchors"], marker="o", label=task)
    ax.set_xlabel("Authorized helper steps")
    ax.set_ylabel("Genuine rescue among failed anchors")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    _save(fig, args.output_dir / "02_genuine_handoff_rescue.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for task in tasks:
        rows = curves[curves["task"] == task].set_index("branch_name").reindex(ORDER)
        ax.plot(rows["mean_helper_steps_actual"], rows["system_success_rate_root"], marker="o", label=task)
        for branch, row in rows.iterrows():
            ax.annotate(LABELS[branch], (row["mean_helper_steps_actual"], row["system_success_rate_root"]), fontsize=8)
    ax.set_xlabel("Actual helper actions")
    ax.set_ylabel("Root-clustered system success")
    ax.set_ylim(-0.03, 1.03)
    ax.text(0.01, 0.02, "Finite shortest values are post-hoc diagnostic upper bounds.", transform=ax.transAxes, fontsize=8)
    ax.legend()
    _save(fig, args.output_dir / "03_success_vs_helper_cost.png")

    categories = ["0", "5", "20", "80", "helper_only", "unresolved_by_this_pair_and_grid", "incomplete"]
    counts = []
    for task in tasks:
        values = opportunities[opportunities["task"] == task]["diagnostic_shortest"].astype(str).value_counts()
        for category in categories:
            counts.append({"task": task, "category": category, "count": int(values.get(category, 0))})
    distribution = pd.DataFrame(counts)
    distribution.to_csv(args.output_dir / "plot_shortest_length_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(1, len(tasks))
    for index, task in enumerate(tasks):
        values = distribution[distribution["task"] == task].set_index("category").reindex(categories)["count"]
        ax.bar(np.arange(len(categories)) + index * width, values, width=width, label=task)
    ax.set_xticks(np.arange(len(categories)) + width * (len(tasks) - 1) / 2, ["0", "5", "20", "80", "helper only", "unresolved", "incomplete"])
    ax.set_ylabel("Anchors")
    ax.legend()
    _save(fig, args.output_dir / "04_shortest_effective_length.png")


if __name__ == "__main__":
    main()
