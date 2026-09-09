"""Create and validate the strict lightweight exit-probe archive."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from recovery_handback.common import sha256_file


ALLOWLIST = (
    "config/exit_probe.json",
    "config/protocol.draft.json",
    "config/frozen_protocol.json",
    "config/test_seed_manifest.json",
    "config/frozen.sha256",
    "config/source_commit.txt",
    "assets/assets.json",
    "assets/policy_pair_square.json",
    "assets/resolved_inputs.json",
    "roots/hb3p_exit_probe/summary.json",
    "roots/hb3p_exit_probe/collected_seed_manifest.json",
    "metrics/exit_probe/summary.json",
    "metrics/exit_probe/decision.json",
    "metrics/exit_probe/prefix_checks.csv",
    "metrics/exit_probe/pair_cases.csv",
    "metrics/exit_probe/oracle_rows.csv",
    "report/HB3P_EXIT_PROBE.md",
    "report/LOCAL_ONLY_ARTIFACTS.md",
)


def _local_only(root: Path) -> str:
    lines = [
        "# Local-Only Artifacts", "",
        "The following runtime assets remain on the experiment machine and are excluded from the lightweight package.", "",
        "| Path | Files | Bytes | Reason |", "|---|---:|---:|---|",
    ]
    for relative, reason in (
        ("roots/hb3p_exit_probe", "canonical payloads, images, baseline trajectories, and Parquet"),
        ("episodes", "complete online NPZ trajectories and decision traces"),
        ("ipc", "ephemeral local inference queue"),
        ("logs", "complete runtime logs"),
    ):
        directory = root / relative
        files = [path for path in directory.rglob("*") if path.is_file()] if directory.is_dir() else []
        lines.append(f"| `{directory.resolve()}` | {len(files)} | {sum(path.stat().st_size for path in files)} | {reason} |")
    lines.extend(["", "Checkpoint identities and required hashes are recorded in the assets and frozen protocol files; no weights are included.", ""])
    return "\n".join(lines)


def package(experiment_root: Path, code_root: Path, output: Path) -> dict:
    root = experiment_root.resolve()
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
        raise FileNotFoundError(f"required exit-probe artifacts missing: {missing}")
    for path in sorted((code_root / "recovery_handback/hb3p").iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".sh"}:
            files.append((path, f"code/{path.name}"))
    protocol = json.loads((root / "config/frozen_protocol.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "metrics/exit_probe/decision.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "hb3p_exit_probe_lightweight_package_v1",
        "source_ref": protocol["source_ref"],
        "code_commit": protocol["code_commit"],
        "decision": decision["decision"],
        "included_files": [name for _, name in files],
        "exclusions": ["model weights", "HDF5", "NPZ trajectories", "Parquet", "videos", "logs", "IPC payloads"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(output) as archive:
        error = archive.testzip()
        names = set(archive.namelist())
    if error is not None or any(relative not in names for relative in ALLOWLIST):
        raise RuntimeError(f"exit-probe ZIP validation failed: bad={error}")
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
