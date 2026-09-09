"""Collect roots for HB2 roles while reusing the frozen HB1 implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import read_table, write_table
from recovery_handback.data.collect_roots import collect
from recovery_handback.data.select_anchors import select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.role not in config["roles"] or not args.role.startswith("hb2_"):
        raise ValueError(f"invalid HB2 role: {args.role}")
    assets_path = Path(config["assets_file"])
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    pair_path = Path(config["hb2"]["selected_policy_pair_path"])
    collect(config, assets, "square", args.role, pair_path, args.output_dir,
            start_index=args.start_index, count=args.count)
    anchors_path = args.output_dir / "anchors.parquet"
    select(config, "square", args.role, args.output_dir / "roots.parquet", anchors_path)
    roots = {row["root_id"]: row for row in read_table(args.output_dir / "roots.parquet")}
    anchors = read_table(anchors_path)
    for row in anchors:
        root = roots[row["root_id"]]
        row["data_role"] = args.role
        row["model_hash"] = root.get("model_hash")
    write_table(anchors, anchors_path)
    write_table(anchors, args.output_dir / "anchors.jsonl")


if __name__ == "__main__":
    main()

