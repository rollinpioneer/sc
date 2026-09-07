"""Create the lightweight, large-artifact-free HB1 delivery archive."""
from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path


EXCLUDED = {".hdf5", ".h5", ".npz", ".pt", ".pth", ".parquet", ".mp4", ".pkl", ".zip"}


def _category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".pth", ".pt", ".zip", ".pkl"}:
        return "model_or_training_state"
    if suffix in {".hdf5", ".h5", ".npz", ".parquet"}:
        return "trajectory_or_dataset"
    if suffix == ".mp4":
        return "video"
    return "large_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    whitelist = [
        "config/hb1.json", "config/runtime.txt", "config/source_commit.txt", "assets/assets.json",
        "assets/policy_pair_can.json", "assets/policy_pair_square.json", "metrics/coverage.json",
        "metrics/curves_by_task.csv", "metrics/curves_by_anchor_time.csv", "metrics/hb1_decision.json",
        "metrics/metric_fixture.json", "metrics/can_minimal_checks.json", "metrics/square_minimal_checks.json",
        "metrics/selected_examples.csv", "report/HB1_REPORT.md",
    ]
    large = []
    for path in args.root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in EXCLUDED or path.stat().st_size > 20 * 1024 * 1024):
            large.append((str(path.resolve()), path.stat().st_size, _category(path)))
    with tempfile.TemporaryDirectory(prefix="hb1-package-") as temp_dir:
        temp = Path(temp_dir)
        readme = temp / "README.md"
        readme.write_text(
            "# HB1 lightweight results\n\nThis archive contains code, configuration, compact metrics, figures, and the report. Model weights, raw trajectories, replay buffers, Parquet tables, and videos are intentionally excluded.\n",
            encoding="utf-8",
        )
        manifest = temp / "large_artifacts.tsv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["path", "bytes", "category"])
            writer.writerows(large)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(readme, "README.md")
            archive.write(manifest, "large_artifacts.tsv")
            for relative in whitelist:
                source = args.root / relative
                if source.is_file() and source.suffix.lower() not in EXCLUDED:
                    archive.write(source, relative)
            for source in sorted(args.root.glob("figures/*.png")):
                archive.write(source, str(source.relative_to(args.root)))
            for source in sorted(args.code_root.rglob("*.py")):
                archive.write(source, str(Path("code/recovery_handback") / source.relative_to(args.code_root)))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
