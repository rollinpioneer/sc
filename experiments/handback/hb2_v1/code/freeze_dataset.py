"""Freeze HB2 train/validation groups and evaluate the one-time data gate."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_json, write_table
from recovery_handback.hb2.labels import rescue_root_counts


def _read(directory: Path, name: str) -> list[dict]:
    return read_table(directory / name)


def _globalize(anchor_rows: list[dict], handoff_rows: list[dict]) -> None:
    by_identity: dict[tuple[str, str], str] = {}
    by_root: dict[str, str] = {}
    for row in anchor_rows:
        identity = (str(row.get("model_hash") or "unknown"), str(row.get("initial_state_hash")))
        canonical = by_identity.setdefault(identity, f"stat:{sha256_json(identity)[:20]}")
        by_root[str(row["root_id"])] = canonical
        row["stat_group_id"] = canonical
    for row in handoff_rows:
        row["stat_group_id"] = by_root.get(str(row["root_id"]), row.get("stat_group_id"))


def _role_counts(anchor_rows: list[dict], handoff_rows: list[dict]) -> list[dict]:
    result = []
    for role in sorted({str(row["data_role"]) for row in anchor_rows}):
        anchors = [row for row in anchor_rows if row["data_role"] == role]
        handoffs = [row for row in handoff_rows if row["data_role"] == role and bool(row.get("eligible"))]
        roots = {row["stat_group_id"] for row in anchors if bool(row.get("complete_pair"))}
        result.append({
            "data_role": role,
            "valid_roots": len(roots),
            "complete_anchors": sum(bool(row.get("complete_pair")) for row in anchors),
            "eligible_handoff_examples": len(handoffs),
            "genuine_handoff_examples": sum(int(row.get("y_strict_genuine", 0)) for row in handoffs),
            **rescue_root_counts(anchors),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-extra", type=Path)
    parser.add_argument("--validation-extra", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sources = [(args.legacy, "legacy_train", "train"), (args.train, "hb2_train", "train"),
               (args.validation, "hb2_val", "validation")]
    if args.train_extra:
        sources.append((args.train_extra, "hb2_train_extra", "train"))
    if args.validation_extra:
        sources.append((args.validation_extra, "hb2_val_extra", "validation"))
    anchor_rows, handoff_rows = [], []
    for directory, role, split in sources:
        current_anchors = _read(directory, "anchor_examples.parquet")
        current_handoffs = _read(directory, "handoff_examples.parquet")
        for row in current_anchors:
            row["data_role"] = role
            row["split"] = split
        for row in current_handoffs:
            row["data_role"] = role
            row["split"] = split
        anchor_rows.extend(current_anchors)
        handoff_rows.extend(current_handoffs)
    _globalize(anchor_rows, handoff_rows)

    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in anchor_rows:
        split_groups[str(row["split"])].add(str(row["stat_group_id"]))
    overlap = sorted(split_groups["train"] & split_groups["validation"])
    if overlap:
        raise ValueError(f"train/validation global root overlap: {overlap[:10]}")

    train_counts = rescue_root_counts(row for row in anchor_rows if row["split"] == "train")
    validation_counts = rescue_root_counts(row for row in anchor_rows if row["split"] == "validation")
    gate = config["hb2"]["data_gate"]
    failures = []
    for split, counts, positive_key, negative_key in (
        ("train", train_counts, "train_rescue_positive_roots", "train_rescue_negative_roots"),
        ("validation", validation_counts, "validation_rescue_positive_roots", "validation_rescue_negative_roots"),
    ):
        if counts["rescue_positive_roots"] < int(gate[positive_key]):
            failures.append(f"{split}_rescue_positive_roots")
        if counts["rescue_negative_roots"] < int(gate[negative_key]):
            failures.append(f"{split}_rescue_negative_roots")
    expansion_used = bool(args.train_extra or args.validation_extra)
    sufficiency = {
        "schema_version": "hb2_data_sufficiency_v1",
        "train": train_counts,
        "validation": validation_counts,
        "failed_gates": failures,
        "single_allowed_expansion_used": expansion_used,
        "expansion_required": bool(failures) and not expansion_used,
        "sufficient": not failures,
        "status": "READY_FEATURES" if not failures else ("EXPANSION_REQUIRED" if not expansion_used else "HOLD_HB2_DATA"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(anchor_rows, args.output_dir / "anchor_examples.parquet")
    write_table(handoff_rows, args.output_dir / "handoff_examples.parquet")
    counts = _role_counts(anchor_rows, handoff_rows)
    with (args.output_dir / "label_counts_by_role.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]) if counts else ["data_role"])
        writer.writeheader()
        writer.writerows(counts)
    manifest = {
        "schema_version": "hb2_split_manifest_v1",
        "semantic_pair_id": config["hb2"]["semantic_pair_id"],
        "roles": {role: {"split": split, "path": str(directory.resolve())}
                  for directory, role, split in sources},
        "train_stat_group_ids": sorted(split_groups["train"]),
        "validation_stat_group_ids": sorted(split_groups["validation"]),
        "test_registered_but_unread": config["roles"]["hb2_test"],
        "anchor_examples": len(anchor_rows),
        "eligible_handoff_examples": sum(bool(row.get("eligible")) for row in handoff_rows),
    }
    atomic_json_dump(manifest, args.output_dir / "split_manifest.json")
    atomic_json_dump(sufficiency, args.output_dir / "data_sufficiency.json")
    print(json.dumps(sufficiency, indent=2))


if __name__ == "__main__":
    main()
