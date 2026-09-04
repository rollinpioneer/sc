"""Create a clearly non-final lightweight Stage 3 progress snapshot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


FORMAL_LOGS = (
    "S3E-B-build-labels.log",
    "S3E-B-feature-smoke.log",
    "S3E-B-feature-full.log",
    "S3E-B-normalizer.log",
    "S3E-C-build-adapter.log",
    "S3E-C-evidence-smoke.log",
    "S3E-C-evidence-full.log",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--evidence-pid", type=int, default=0)
    parser.add_argument("--oracle-pid", type=int, default=0)
    parser.add_argument("--log-lines", type=int, default=40)
    args = parser.parse_args()
    root = args.root.resolve()
    report_dir = args.report_dir.resolve()
    log_tail_dir = report_dir / "log_tails"
    log_tail_dir.mkdir(parents=True, exist_ok=True)
    log_status = {}
    for name in FORMAL_LOGS:
        source = root / "logs" / name
        destination = log_tail_dir / name
        if source.exists():
            lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
            destination.write_text("\n".join(lines[-args.log_lines:]) + ("\n" if lines else ""), encoding="utf-8")
            log_status[name] = {"available": True, "tail_path": str(destination.relative_to(root)), "source_size_bytes": source.stat().st_size}
        else:
            log_status[name] = {"available": False}
    snapshot = {
        "schema": "stage3_v02r_progress_snapshot_v1",
        "snapshot_kind": "in_progress_not_final",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": (root / "config" / "base_commit.txt").read_text(encoding="utf-8").strip(),
        "completed_stages": ["3E-A frozen inputs and confirmation gate", "3E-B labels, features, and train-only verifier normalization", "3E-C frozen evidence and full/action proposer transfer"],
        "in_progress_stage": "3E-D train candidate-level long counterfactual oracle (16 fixed shards)",
        "evidence_export_pid": args.evidence_pid,
        "candidate_oracle_pid": args.oracle_pid,
        "finalization_state": "No verifier training, validation protocol selection, or blind benchmark has been run.",
        "large_artifact_policy": "HDF5, NPZ, Parquet, checkpoints, FAISS, videos, and JSONL are recorded in the manifest and excluded from the lightweight ZIP.",
        "logs": log_status,
    }
    (report_dir / "progress_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Stage 3 v0.2-R Progress Snapshot",
        "",
        "Status: in progress, not a final Stage 3H delivery.",
        "",
        "Completed: 3E-A frozen-input gate; 3E-B labels, features, and train-only verifier normalization; 3E-C frozen evidence and proposer transfer.",
        "",
        "In progress: 3E-D train candidate-level long counterfactual oracle (16 fixed shards).",
        "",
        "Not run: verifier training, validation protocol selection, blind benchmark generation, and blind test.",
        "",
        "Large artifacts are catalogued in `large_artifact_manifest.json` and intentionally excluded from this ZIP.",
    ]
    (report_dir / "progress_snapshot.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"report_dir": str(report_dir), "log_count": sum(item["available"] for item in log_status.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
