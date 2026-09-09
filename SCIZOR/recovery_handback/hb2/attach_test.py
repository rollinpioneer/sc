"""Attach a post-freeze test role without mutating the frozen train/validation tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_json, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-dataset", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors = read_table(args.test_data / "anchor_examples.parquet")
    handoffs = read_table(args.test_data / "handoff_examples.parquet")
    groups = defaultdict(dict)
    for row in anchors:
        identity = (str(row.get("model_hash") or "unknown"), str(row.get("initial_state_hash")))
        group = f"stat:{sha256_json(identity)[:20]}"
        groups[str(row["root_id"])]["group"] = group
        row["stat_group_id"] = group
        row["split"] = "test"
        row["data_role"] = "hb2_test"
    for row in handoffs:
        row["stat_group_id"] = groups[str(row["root_id"])].get("group", row.get("stat_group_id"))
        row["split"] = "test"
        row["data_role"] = "hb2_test"
    write_table(anchors, args.output_dir / "anchor_examples.parquet")
    write_table(handoffs, args.output_dir / "handoff_examples.parquet")
    manifest = {
        "schema_version": "hb2_test_attached_v1",
        "source_test_data": str(args.test_data.resolve()),
        "frozen_dataset": str(args.frozen_dataset.resolve()),
        "semantic_pair_id": config["hb2"]["semantic_pair_id"],
        "anchors": len(anchors),
        "handoff_examples": len(handoffs),
        "test_roots": len({row["stat_group_id"] for row in anchors}),
        "frozen_train_validation_unchanged": True,
    }
    atomic_json_dump(manifest, args.output_dir / "test_attach_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

