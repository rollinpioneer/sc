"""Package only lightweight HB2 summaries, commands, and source changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _large_local_assets(root: Path) -> list[dict]:
    records = []
    for directory in ("features", "models", "roots", "branches", "predictions"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.stat().st_size < 1024 * 1024:
                continue
            records.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "included_in_lightweight_zip": False,
            })
    return records


def _git_diff(repo: Path, source_ref: str) -> tuple[str, str]:
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--stat", f"{source_ref}..HEAD", "--",
         "SCIZOR/recovery_handback/hb2"],
        text=True,
    )
    names = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-status", f"{source_ref}..HEAD", "--",
         "SCIZOR/recovery_handback/hb2"],
        text=True,
    )
    return diff, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    repo = Path(__file__).resolve().parents[3]
    config = json.loads((root / "config/hb2.json").read_text(encoding="utf-8"))
    files: list[tuple[Path, Path]] = []

    patterns = (
        "config/*.json", "config/*.txt",
        "assets/*.json",
        "data/frozen/*.json", "data/frozen/*.csv",
        "features/*.json", "features/handoff/*.json",
        "metrics/**/*.json", "metrics/**/*.csv",
        "report/*.md",
    )
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files.append((path, path.relative_to(root)))
    code_root = repo / "SCIZOR/recovery_handback/hb2"
    for pattern in ("*.py", "*.json", "*.sh"):
        for path in sorted(code_root.glob(pattern)):
            if path.is_file():
                files.append((path, Path("code") / path.name))

    large_assets = _large_local_assets(root)
    diff_stat, diff_names = _git_diff(repo, str(config["source_ref"]))
    command_files = [
        code_root / "run_data_background.sh",
        code_root / "run_model_background.sh",
    ]
    commands = "# HB2 Execution Commands\n\n" + "\n\n".join(
        f"## {path.name}\n\n```bash\n{path.read_text(encoding='utf-8')}\n```"
        for path in command_files if path.is_file()
    )
    package_manifest = {
        "schema_version": "hb2_lightweight_package_v2",
        "experiment_root": str(root.resolve()),
        "source_ref": config["source_ref"],
        "included_file_count": len(files) + 5,
        "exclusions": [
            "model weights", "feature NPZ caches", "raw roots and branch trajectories",
            "prediction payloads", "images and videos", "complete logs",
        ],
        "local_large_assets": large_assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        seen = set()
        for path, arcname in files:
            name = arcname.as_posix()
            if name in seen:
                continue
            seen.add(name)
            archive.write(path, name)
        archive.writestr("RUN_COMMANDS.md", commands)
        archive.writestr("CODE_DIFF.stat.txt", diff_stat)
        archive.writestr("CODE_DIFF.name-status.txt", diff_names)
        archive.writestr(
            "LOCAL_ASSETS.json",
            json.dumps(large_assets, indent=2, sort_keys=True) + "\n",
        )
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        )
    digest = _sha256(args.output)
    checksum_path = args.output.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "files": package_manifest["included_file_count"],
        "local_large_assets": len(large_assets),
        "sha256": digest,
    }, indent=2))


if __name__ == "__main__":
    main()
