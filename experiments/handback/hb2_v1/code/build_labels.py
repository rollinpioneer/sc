"""Build legacy HB1-R anchor and handoff examples for HB2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump
from recovery_handback.hb2.labels import build_example_tables, write_example_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = json.loads(args.inputs.read_text(encoding="utf-8"))
    entries = resolved["inputs"]
    anchor_rows, handoff_rows, coverage = build_example_tables(
        Path(entries["anchors"]["resolved_path"]),
        Path(entries["branch_records"]["resolved_path"]),
        roots_path=Path(entries["roots"]["resolved_path"]),
        role=args.role,
        semantic_pair_id=resolved["semantic_pair_id"],
        minimum_autonomous_steps=int(config["minimum_autonomous_steps"]),
    )
    write_example_tables(anchor_rows, handoff_rows, args.output_dir)
    atomic_json_dump(coverage, args.output_dir / "coverage.json")
    clarification = {
        "schema_version": "hb2_hb1_count_clarification_v1",
        "role": args.role,
        "success_anchor_rate": coverage["full_completion"]["anchor_rate"],
        "mean_within_root_success_rate": coverage["full_completion"]["mean_within_root_rate"],
        "unique_success_roots": coverage["full_completion"]["unique_helper_completed_roots"],
        "details": coverage["full_completion"],
        "note": "The three quantities use different denominators and are not interchangeable.",
    }
    experiment_root = Path(config["output_root"])
    atomic_json_dump(clarification, experiment_root / "metrics/hb1_count_clarification.json")
    print(json.dumps({"anchor_examples": len(anchor_rows),
                      "eligible_handoff_examples": sum(bool(row["eligible"]) for row in handoff_rows),
                      "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()

