"""Aggregate per-branch NPZ traces into one HDF5 file per root scenario."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.execution.label_reference import BRANCH_SPECS


BRANCH_ORDER = {name: index for index, (name, _length) in enumerate(BRANCH_SPECS)}


def _slug(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")


def _write_root(root_id: str, rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(temp_name, "w") as handle:
            handle.attrs["schema_version"] = "hb1_root_branches_v1"
            handle.attrs["root_id"] = root_id
            for row in sorted(
                rows, key=lambda item: (int(item["anchor_t"]), BRANCH_ORDER[item["branch_name"]])
            ):
                trajectory = Path(row["trajectory_path"])
                if not trajectory.is_file():
                    raise FileNotFoundError(trajectory)
                group = handle.require_group(
                    f"anchors/{int(row['anchor_t']):03d}/{row['branch_name']}"
                )
                group.attrs["anchor_id"] = row["anchor_id"]
                group.attrs["result_json"] = json.dumps(row, sort_keys=True, default=str)
                with np.load(trajectory, allow_pickle=False) as source:
                    for key in ("actions", "base_actions", "helper_mask", "states", "success", "rewards"):
                        group.create_dataset(
                            key, data=np.asarray(source[key]), compression="gzip", shuffle=True
                        )
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_table(args.branch_results)
    by_root: dict[str, list[dict]] = {}
    for row in rows:
        by_root.setdefault(str(row["root_id"]), []).append(row)
    outputs = []
    for root_id, root_rows in sorted(by_root.items()):
        output = args.output_dir / f"{_slug(root_id)}.hdf5"
        _write_root(root_id, root_rows, output)
        outputs.append({"root_id": root_id, "path": str(output.resolve()), "records": len(root_rows)})
    atomic_json_dump(
        {"schema_version": "hb1_root_branch_archives_v1", "roots": outputs},
        args.output_dir / "manifest.json",
    )
    print(json.dumps({"roots": len(outputs), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
