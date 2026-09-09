"""Build a strict lightweight package for the stop/continue pilot."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from recovery_handback.common import sha256_file


ALLOWLIST = (
    "config/stop_continue.json", "config/protocol.draft.json", "config/frozen_protocol.json", "config/frozen.sha256",
    "config/source_commit.txt", "dataset/dataset_manifest.json", "model/training_summary.json", "model/oof_predictions.jsonl",
    "metrics/coverage.json", "metrics/summary.json", "metrics/decision.json", "metrics/methods.csv", "metrics/prefix_checks.csv",
    "report/HB3P_STOP_CONTINUE.md", "report/LOCAL_ONLY_ARTIFACTS.md",
)


def package(experiment_root: Path, code_root: Path, output: Path) -> dict:
    root = Path(experiment_root)
    local_only = root / "report/LOCAL_ONLY_ARTIFACTS.md"
    local_only.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Local-Only Artifacts", "", "Runtime assets excluded from the lightweight package:", "", "- test-root payloads, rollouts, images, and full trajectories", "- trained model checkpoint and normalizer", "- full logs and transient IPC", "", "Only hashes, protocol, metrics, report, and source code are included.", ""]
    local_only.write_text("\n".join(lines), encoding="utf-8")
    files, missing = [], []
    for relative in ALLOWLIST:
        path = root / relative
        (files if path.is_file() else missing).append((path, relative) if path.is_file() else relative)
    if missing:
        raise FileNotFoundError(f"missing package artifacts: {missing}")
    package_code = Path(code_root) / "recovery_handback/hb3p_stop"
    for path in sorted(package_code.iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".sh"}:
            files.append((path, f"code/{path.name}"))
    protocol = json.loads((root / "config/frozen_protocol.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "hb3p_stop_continue_lightweight_package_v1",
        "source_probe_protocol_sha256": protocol["source_probe_protocol_sha256"],
        "frozen_code_commit": protocol["code_commit"],
        "included_files": [name for _, name in files],
        "exclusions": ["HDF5", "NPZ trajectories", "Parquet", "model weights", "images", "full logs", "IPC payloads"],
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("package ZIP integrity check failed")
        names = set(archive.namelist())
    if any(relative not in names for relative in ALLOWLIST):
        raise RuntimeError("package ZIP allowlist validation failed")
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {"output": str(output.resolve()), "files": len(files) + 1, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

