"""Create and validate the strict lightweight HB3-P result archive."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from recovery_handback.common import sha256_file


ALLOWLIST = (
    "config/hb3p.json",
    "config/source_commit.txt",
    "config/frozen_protocol.json",
    "config/test_seed_manifest.json",
    "config/frozen.sha256",
    "assets/resolved_inputs.json",
    "assets/policy_pair_square.json",
    "metrics/clarification/ranking_metrics.json",
    "metrics/clarification/HB2_REPORT_NOTES.md",
    "metrics/development/fixed_schedule_candidates.csv",
    "metrics/development/m0_calibrated_schedule.json",
    "metrics/development/method_aliases.json",
    "metrics/development/root_episode_metrics.csv",
    "metrics/pilot/summary.json",
    "metrics/test/coverage.json",
    "metrics/test/methods.csv",
    "metrics/test/paired_comparisons.json",
    "metrics/test/paired_outcome_counts.csv",
    "metrics/test/interference.csv",
    "metrics/test/costs.csv",
    "metrics/test/decision_time_counts.csv",
    "metrics/test/equivalent_methods.json",
    "metrics/hb3p_decision.json",
    "report/HB3P_REPORT.md",
    "report/LOCAL_ONLY_ARTIFACTS.md",
    "report/development_cases.json",
)


def _local_only(root: Path) -> str:
    lines = [
        "# Local-Only Artifacts", "",
        "The following runtime assets remain on the experiment machine and are excluded from the lightweight package.", "",
        "| Path | Files | Bytes | Reason |", "|---|---:|---:|---|",
    ]
    for relative, reason in (
        ("roots/hb3p_test", "canonical payloads, images, baseline trajectories, and Parquet"),
        ("episodes/test", "complete online NPZ trajectories and decision traces"),
        ("ipc", "ephemeral local inference queue"),
        ("logs", "complete runtime logs"),
    ):
        directory = root / relative
        files = [path for path in directory.rglob("*") if path.is_file()] if directory.is_dir() else []
        lines.append(f"| `{directory.resolve()}` | {len(files)} | {sum(path.stat().st_size for path in files)} | {reason} |")
    lines.extend([
        "", "Checkpoint identities and required hashes are recorded in `assets/resolved_inputs.json` and `config/frozen_protocol.json`; no weights are included.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    local_only = root / "report/LOCAL_ONLY_ARTIFACTS.md"
    local_only.parent.mkdir(parents=True, exist_ok=True)
    local_only.write_text(_local_only(root), encoding="utf-8")
    files = []
    missing = []
    for relative in ALLOWLIST:
        path = root / relative
        if path.is_file():
            files.append((path, relative))
        else:
            missing.append(relative)
    if missing:
        raise FileNotFoundError(f"required lightweight artifacts missing: {missing}")
    code_root = Path(__file__).resolve().parent
    for path in sorted(code_root.iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".sh"}:
            files.append((path, f"code/{path.name}"))
    manifest = {
        "schema_version": "hb3p_lightweight_package_v1",
        "source_ref": json.loads((root / "config/frozen_protocol.json").read_text())["source_ref"],
        "included_files": [name for _, name in files],
        "exclusions": [
            "model weights", "DINO caches", "HDF5", "NPZ trajectories", "Parquet",
            "videos", "full logs", "IPC payloads",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        error = archive.testzip()
        names = set(archive.namelist())
    if error is not None or any(relative not in names for relative in ALLOWLIST):
        raise RuntimeError(f"lightweight ZIP validation failed: bad={error}")
    digest = sha256_file(args.output)
    checksum = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "files": len(files) + 1, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
