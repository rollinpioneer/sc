"""Create the bounded, large-artifact-free HB1-R delivery archive."""
from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path


EXCLUDED_SUFFIXES = {
    ".avi", ".ckpt", ".h5", ".hdf5", ".log", ".mkv", ".mp4",
    ".npz", ".parquet", ".pkl", ".pt", ".pth", ".zip",
}


def _category(path: Path) -> str:
    if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".zip", ".pkl"}:
        return "model_or_training_state"
    if path.suffix.lower() in {".h5", ".hdf5", ".npz", ".parquet"}:
        return "trajectory_or_dataset"
    if path.suffix.lower() in {".avi", ".mkv", ".mp4"}:
        return "video"
    if path.suffix.lower() == ".log":
        return "full_log"
    return "large_other"


def _eligible(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() not in EXCLUDED_SUFFIXES


def _collect(root: Path) -> list[Path]:
    fixed = [
        root / "config/hb1_repair.json",
        root / "config/runtime.txt",
        root / "config/source_commit.txt",
        root / "config/frozen_inputs.sha256",
        root / "square/teacher/evaluation/qualification/summary.json",
        root / "square/repair_rl/index/curriculum_summary.json",
        root / "square/repair_rl/qualification/summary.json",
        root / "report/HB1_R_REPORT.md",
    ]
    patterns = [
        "assets/*.json",
        "square/teacher/**/*manifest*.json",
        "can/diagnostic/**/*.json",
        "can/distill/**/*manifest*.json",
        "can/visual/evaluation/*summary*.json",
        "metrics/*.json",
        "metrics/*.csv",
    ]
    paths = {path for path in fixed if _eligible(path)}
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if _eligible(path))
    return sorted(paths, key=lambda path: str(path.relative_to(root)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    code_root = args.code_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    large = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == output:
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            large.append((str(path.resolve()), path.stat().st_size, _category(path)))

    with tempfile.TemporaryDirectory(prefix="hb1-r-package-") as temp_dir:
        temp = Path(temp_dir)
        readme = temp / "README.md"
        readme.write_text(
            "# HB1-R lightweight results\n\n"
            "This archive contains compact configuration, capability and formal-probe summaries, "
            "the report, and recovery_handback Python source. Model weights, training state, raw "
            "datasets, trajectories, Parquet tables, videos, and full logs are excluded.\n",
            encoding="utf-8",
        )
        manifest = temp / "excluded_large_artifacts.tsv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["path", "bytes", "category"])
            writer.writerows(sorted(large))
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(readme, "README.md")
            archive.write(manifest, "excluded_large_artifacts.tsv")
            for source in _collect(root):
                archive.write(source, str(source.relative_to(root)))
            for source in sorted(code_root.rglob("*.py")):
                archive.write(
                    source,
                    str(Path("code/recovery_handback") / source.relative_to(code_root)),
                )
    print(output)


if __name__ == "__main__":
    main()
