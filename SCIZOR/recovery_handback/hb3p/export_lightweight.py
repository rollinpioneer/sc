"""Export only reviewable HB3-P summaries and the validated lightweight ZIP."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from recovery_handback.common import sha256_file
from recovery_handback.hb3p.package import ALLOWLIST


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    destination = args.destination.resolve()
    files = list(ALLOWLIST) + [
        "package/HB3P_results_lightweight.zip",
        "package/HB3P_results_lightweight.zip.sha256",
    ]
    copied = []
    for relative in files:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"export hash mismatch: {relative}")
        copied.append(relative)
    readme = destination / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# HB3-P v1\n\n"
        "- [Report](report/HB3P_REPORT.md)\n"
        "- [Frozen protocol](config/frozen_protocol.json)\n"
        "- [Decision](metrics/hb3p_decision.json)\n"
        "- [Method metrics](metrics/test/methods.csv)\n"
        "- [Paired comparisons](metrics/test/paired_comparisons.json)\n"
        "- [Exit diagnosis](metrics/diagnosis/HB3P_EXIT_DIAGNOSIS.md)\n"
        "- [Exit route decision](metrics/diagnosis/exit_diagnosis_decision.json)\n"
        "- [Lightweight results ZIP](package/HB3P_results_lightweight.zip)\n"
        "- [ZIP SHA256](package/HB3P_results_lightweight.zip.sha256)\n\n"
        "Large trajectories, model weights, feature caches, Parquet files, and logs remain local.\n",
        encoding="utf-8",
    )
    copied.append("README.md")
    manifest = {
        "schema_version": "hb3p_lightweight_export_v1",
        "source_root": str(root),
        "destination": str(destination),
        "files": copied,
    }
    (destination / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"destination": str(destination), "files": len(copied) + 1}, indent=2))


if __name__ == "__main__":
    main()
