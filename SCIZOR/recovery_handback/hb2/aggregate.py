"""Aggregate one HB2 role across independent branch shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump
from recovery_handback.hb2.labels import build_example_tables, write_example_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    roots_path = args.anchors.with_name("roots.parquet")
    anchor_rows, handoff_rows, coverage = build_example_tables(
        args.anchors,
        args.branches_root,
        roots_path=roots_path if roots_path.is_file() else None,
        role=args.role,
        semantic_pair_id=config["hb2"]["semantic_pair_id"],
        minimum_autonomous_steps=int(config["minimum_autonomous_steps"]),
    )
    write_example_tables(anchor_rows, handoff_rows, args.output_dir)
    atomic_json_dump(coverage, args.output_dir / "coverage.json")
    print(json.dumps({"role": args.role, "anchor_examples": len(anchor_rows),
                      "handoff_examples": len(handoff_rows), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()

