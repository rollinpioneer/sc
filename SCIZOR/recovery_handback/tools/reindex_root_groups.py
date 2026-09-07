"""Recompute canonical state hashes and duplicate-scene statistical groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, sha256_array, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-dir", type=Path, required=True)
    parser.add_argument("--anchors", type=Path)
    args = parser.parse_args()
    rows = read_table(args.roots_dir)
    group_ids = {}
    root_to_group = {}
    for row in sorted(rows, key=lambda item: (item["task"], item["role"], int(item["seed"]))):
        with np.load(row["payload_path"], allow_pickle=False) as handle:
            row["initial_state_hash"] = sha256_array(
                np.asarray(handle["states"], dtype=np.float64)
            )
        key = (row["model_hash"], row["initial_state_hash"])
        row["stat_group_id"] = group_ids.setdefault(key, row["root_id"])
        root_to_group[row["root_id"]] = row["stat_group_id"]
    write_table(rows, args.roots_dir / "roots.parquet")
    write_table(rows, args.roots_dir / "roots.jsonl")
    summary_path = args.roots_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary["rows"] = rows
    summary["duplicate_scene_roots"] = len(rows) - len(group_ids)
    atomic_json_dump(summary, summary_path)
    if args.anchors and args.anchors.is_file():
        anchors = read_table(args.anchors)
        root_hash = {row["root_id"]: row["initial_state_hash"] for row in rows}
        for anchor in anchors:
            anchor["stat_group_id"] = root_to_group[anchor["root_id"]]
            anchor["initial_state_hash"] = root_hash[anchor["root_id"]]
        write_table(anchors, args.anchors)
    print(json.dumps({
        "roots": len(rows), "stat_groups": len(group_ids),
        "duplicate_scene_roots": len(rows) - len(group_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
