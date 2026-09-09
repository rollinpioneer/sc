"""Copy only the lightweight pilot artifacts into the Git worktree."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(experiment_root: Path, destination: Path) -> dict:
    root, destination = Path(experiment_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for relative in ("config/stop_continue.json", "config/protocol.draft.json", "config/frozen_protocol.json", "config/frozen.sha256", "config/source_commit.txt", "dataset/dataset_manifest.json", "model/training_summary.json", "model/oof_predictions.jsonl", "metrics/coverage.json", "metrics/summary.json", "metrics/decision.json", "metrics/methods.csv", "metrics/prefix_checks.csv", "metrics/branch_parity.csv", "report/HB3P_STOP_CONTINUE.md", "report/LOCAL_ONLY_ARTIFACTS.md", "package/HB3P_stop_continue_lightweight.zip", "package/HB3P_stop_continue_lightweight.zip.sha256"):
        source = root / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return {"destination": str(destination.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(export(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
