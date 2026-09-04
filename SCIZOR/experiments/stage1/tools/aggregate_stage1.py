#!/usr/bin/env python3
"""Aggregate already-produced Stage-1 artifacts; never fabricate metrics."""
import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--experiment-dir", required=True); p.add_argument("--report", required=True)
    a = p.parse_args(); root = Path(a.experiment_dir)
    manifest, rollout, cost = (read_json(root / x) for x in ("run_manifest.json", "rollout/summary.json", "metrics/cost.json"))
    score_file, selection_file = root / "influence/influence_scores.csv", root / "selection/selected_clusters.csv"
    scores = list(csv.DictReader(score_file.open())) if score_file.is_file() else []
    selection = list(csv.DictReader(selection_file.open())) if selection_file.is_file() else []
    result = {"experiment_id": (manifest or {}).get("experiment_id"), "status": (manifest or {}).get("status", "missing_manifest"),
              "cluster_count": len(scores), "selected_count": len(selection), "rollout": rollout, "cost": cost,
              "artifacts": {"influence_scores": str(score_file), "selected_clusters": str(selection_file),
                            "rollout_summary": str(root / "rollout/summary.json"), "cost": str(root / "metrics/cost.json")}}
    metrics = root / "metrics/baseline_metrics.json"; metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Stage 1 baseline summary", "", f"- experiment_id: `{result['experiment_id']}`", f"- status: `{result['status']}`",
             f"- influence rows: {len(scores)}", f"- selected clusters: {len(selection)}",
             f"- rollout: `{rollout if rollout is not None else 'not produced'}`", f"- cost: `{cost if cost is not None else 'not produced'}`",
             "- Ready for Stage 1.3: no — a completed formal DataMIL run is required.", ""]
    report = Path(a.report); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines))


if __name__ == "__main__": main()
