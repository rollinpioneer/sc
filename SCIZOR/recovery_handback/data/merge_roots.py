"""Merge non-overlapping HB1 root-collection shards into one authoritative table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, sha256_array, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    rows = []
    shard_paths = []
    for shard in sorted(path for path in args.shards_root.iterdir() if path.is_dir()):
        table = shard / "roots.parquet"
        if not table.is_file():
            raise RuntimeError(f"incomplete root shard: {shard}")
        shard_rows = read_table(table)
        rows.extend(shard_rows)
        shard_paths.append(str(shard.resolve()))
    rows.sort(key=lambda row: (row["task"], row["role"], int(row["seed"])))
    root_ids = [row["root_id"] for row in rows]
    if len(root_ids) != len(set(root_ids)):
        raise RuntimeError("duplicate root_id across root shards")
    if len(rows) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} roots, found {len(rows)}")

    group_ids = {}
    for row in rows:
        with np.load(row["payload_path"], allow_pickle=False) as handle:
            row["initial_state_hash"] = sha256_array(
                np.asarray(handle["states"], dtype=np.float64)
            )
        key = (row["model_hash"], row["initial_state_hash"])
        row["stat_group_id"] = group_ids.setdefault(key, row["root_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, args.output_dir / "roots.parquet")
    write_table(rows, args.output_dir / "roots.jsonl")
    atomic_json_dump({
        "schema_version": "hb1_roots_summary_v1",
        "task": rows[0]["task"] if rows else None,
        "role": rows[0]["role"] if rows else None,
        "roots": len(rows),
        "baseline_successes": sum(bool(row.get("baseline_success")) for row in rows),
        "exception_roots": sum(bool(row.get("exception_reason")) for row in rows),
        "source_shards": shard_paths,
        "rows": rows,
    }, args.output_dir / "summary.json")
    print(json.dumps({
        "roots": len(rows),
        "baseline_successes": sum(bool(row.get("baseline_success")) for row in rows),
        "exception_roots": sum(bool(row.get("exception_reason")) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
