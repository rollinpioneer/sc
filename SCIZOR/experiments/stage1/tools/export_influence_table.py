#!/usr/bin/env python3
"""Standardize real DataMIL score exports without recomputing scores."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import yaml

REQUIRED = ["candidate_id", "trajectory_id", "cluster_id", "influence_score"]
OUTPUT = ["experiment_id", "benchmark", "task_id", "candidate_id", "trajectory_id", "cluster_id",
          "start_step", "end_step", "num_frames", "duration", "influence_score", "rank", "selected",
          "target_demo_count", "target_demo_ids", "data_seed", "influence_seed", "source_checkpoint"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata")
    p.add_argument("--ascending", action="store_true", help="Use only if the original DataMIL code defines lower as better.")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    with Path(args.raw_input).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [name for name in REQUIRED if any(not row.get(name) for row in rows)]
    if missing: raise SystemExit(f"raw influence table is missing required columns/values: {missing}")
    rows.sort(key=lambda row: float(row["influence_score"]), reverse=not args.ascending)
    selected_ids = set()
    budget = config.get("selection", {}).get("budget")
    if isinstance(budget, int): selected_ids = {row["cluster_id"] for row in rows[:budget]}
    task = (config.get("benchmark", {}).get("task_ids") or [""])[0]
    data = config.get("data", {})
    for rank, row in enumerate(rows, 1):
        row.update({"experiment_id": args.experiment_id, "benchmark": config.get("benchmark", {}).get("name", ""),
                    "task_id": row.get("task_id", task), "rank": rank,
                    "selected": str(row["cluster_id"] in selected_ids).lower(),
                    "target_demo_count": data.get("target_demo_count", ""),
                    "target_demo_ids": json.dumps(data.get("target_demo_ids", [])),
                    "data_seed": data.get("data_seed", ""),
                    "influence_seed": config.get("train", {}).get("train_seed", "")})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT, extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({name: row.get(name, "") for name in OUTPUT})
    metadata = Path(args.metadata) if args.metadata else out.with_name("influence_metadata.json")
    metadata.write_text(json.dumps({"score_direction": "ascending" if args.ascending else "descending",
        "raw_input": str(Path(args.raw_input).resolve()), "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cluster_count": len({row["cluster_id"] for row in rows}), "missing_score_count": 0}, indent=2) + "\n")


if __name__ == "__main__": main()
