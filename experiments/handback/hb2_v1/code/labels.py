"""Shared HB-2 branch aggregation and label construction."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from recovery_handback.common import read_table, sha256_json, write_table
from recovery_handback.execution.label_reference import apply_success_labels, diagnostic_shortest
from recovery_handback.hb2.reference_labels import (
    BRANCH_NAMES,
    FINITE_NAMES,
    anchor_outcomes,
    full_completion_summary,
)


def missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def load_branch_rows(source: Path) -> list[dict]:
    source = Path(source)
    if source.is_file():
        return read_table(source)
    paths = []
    if (source / "records").is_dir():
        paths.extend((source / "records").glob("*/*.json"))
    else:
        paths.extend(source.glob("shard_*/records/*/*.json"))
        paths.extend(source.glob("*/records/*/*.json"))
    rows = []
    for path in sorted(set(paths)):
        if path.stem not in BRANCH_NAMES:
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        parquet = sorted(source.glob("**/branch_results*.parquet"))
        for path in parquet:
            rows.extend(read_table(path))
    return rows


def _deduplicate(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for source in rows:
        row = dict(source)
        name = row.get("branch_name")
        if name not in BRANCH_NAMES:
            continue
        key = (str(row.get("anchor_id")), str(name))
        if key in indexed and sha256_json(indexed[key]) != sha256_json(row):
            raise ValueError(f"conflicting duplicate branch record: {key}")
        indexed[key] = row
    return indexed


def _model_hash(anchor: dict, root_by_id: dict[str, dict]) -> str | None:
    root = root_by_id.get(str(anchor.get("root_id")), {})
    return anchor.get("model_hash") or root.get("model_hash")


def build_example_tables(
    anchors_path: Path,
    branches_source: Path,
    *,
    role: str,
    semantic_pair_id: str,
    minimum_autonomous_steps: int,
    roots_path: Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    anchors = read_table(anchors_path)
    roots = read_table(roots_path) if roots_path and Path(roots_path).is_file() else []
    root_by_id = {str(row["root_id"]): row for row in roots}
    indexed = _deduplicate(load_branch_rows(branches_source))
    anchor_examples: list[dict] = []
    handoff_examples: list[dict] = []
    all_complete_branch_rows: list[dict] = []
    missing_keys: list[list[str]] = []
    invalid_keys: list[list[str]] = []
    for anchor in anchors:
        anchor_id = str(anchor["anchor_id"])
        branch_rows = []
        source_paths = {}
        for name in BRANCH_NAMES:
            row = indexed.get((anchor_id, name))
            if row is None:
                missing_keys.append([anchor_id, name])
                continue
            row = dict(row)
            if bool(row.get("engineering_ok")):
                apply_success_labels(row, minimum_autonomous_steps)
                branch_rows.append(row)
                all_complete_branch_rows.append(row)
            else:
                invalid_keys.append([anchor_id, name])
            source_paths[name] = row.get("trajectory_path")
        complete = len(branch_rows) == len(BRANCH_NAMES)
        base = {
            "example_id": f"F:{anchor_id}",
            "task": anchor.get("task", "square"),
            "data_role": role,
            "root_id": anchor.get("root_id"),
            "stat_group_id": anchor.get("stat_group_id", anchor.get("root_id")),
            "anchor_id": anchor_id,
            "anchor_t": int(anchor["anchor_t"]),
            "semantic_pair_id": semantic_pair_id,
            "remaining_steps": int(anchor.get("remaining_steps", 0)),
            "anchor_history_path": anchor.get("anchor_history_path"),
            "rollout_path": anchor.get("rollout_path"),
            "initial_state_hash": anchor.get("initial_state_hash"),
            "model_hash": _model_hash(anchor, root_by_id),
            "complete_pair": complete,
            "engineering_invalid": bool(invalid_keys and any(k[0] == anchor_id for k in invalid_keys)),
            "source_record_paths": source_paths,
        }
        if complete:
            outcomes = anchor_outcomes(branch_rows)
            base.update(outcomes)
            by_name = {row["branch_name"]: row for row in branch_rows}
            for name in BRANCH_NAMES:
                base[f"helper_steps_{name}"] = int(by_name[name].get("helper_steps_actual") or 0)
                base[f"system_success_{name}"] = int(bool(by_name[name].get("system_success")))
                base[f"helper_completed_{name}"] = int(bool(by_name[name].get("helper_completed_task")))
                base[f"changed_action_steps_{name}"] = int(by_name[name].get("changed_action_steps") or 0)
                base[f"repair_calls_after_handoff_{name}"] = int(
                    by_name[name].get("repair_calls_after_handoff") or 0
                )
                base[f"handoff_executed_{name}"] = int(bool(by_name[name].get("handoff_executed")))
            base["diagnostic_shortest"] = diagnostic_shortest(by_name)
        anchor_examples.append(base)

        for row in branch_rows:
            name = row["branch_name"]
            if name not in FINITE_NAMES:
                continue
            handoff_t = row.get("handoff_state_index")
            remaining = None if missing(handoff_t) else int(row["horizon_steps"]) - int(handoff_t)
            reasons = []
            if not bool(row.get("handoff_executed")):
                reasons.append("handoff_not_executed")
            if bool(row.get("success_seen_under_helper")):
                reasons.append("success_seen_under_helper")
            if remaining is None or remaining < 20:
                reasons.append("insufficient_remaining_steps")
            history_path = row.get("handoff_history_path")
            if not history_path or not Path(history_path).is_file():
                reasons.append("missing_handoff_history")
            eligible = not reasons
            system = int(bool(row.get("system_success")))
            genuine = int(bool(row.get("genuine_handoff_success")))
            label = 2 if genuine else (1 if system else 0)
            handoff_examples.append({
                "example_id": f"H:{anchor_id}:{name}",
                "task": anchor.get("task", "square"),
                "data_role": role,
                "root_id": anchor.get("root_id"),
                "stat_group_id": anchor.get("stat_group_id", anchor.get("root_id")),
                "anchor_id": anchor_id,
                "branch_name": name,
                "handoff_t": None if missing(handoff_t) else int(handoff_t),
                "helper_length": int(name[1:]),
                "remaining_steps": remaining,
                "handoff_history_path": history_path,
                "y_complete_after_handoff": system,
                "y_strict_genuine": genuine,
                "handoff_category": label,
                "eligible": eligible,
                "eligibility_reason": "eligible" if eligible else ",".join(reasons),
                "semantic_pair_id": semantic_pair_id,
                "initial_state_hash": anchor.get("initial_state_hash"),
                "model_hash": _model_hash(anchor, root_by_id),
            })
    coverage = {
        "authoritative_anchors": len(anchors),
        "expected_branches": len(anchors) * len(BRANCH_NAMES),
        "present_unique_branches": len(indexed),
        "complete_anchors": sum(bool(row["complete_pair"]) for row in anchor_examples),
        "missing_keys": missing_keys,
        "invalid_keys": invalid_keys,
    }
    coverage["full_branch_coverage"] = (
        coverage["complete_anchors"] == coverage["authoritative_anchors"]
        and not missing_keys and not invalid_keys
    )
    coverage["full_completion"] = full_completion_summary(all_complete_branch_rows)
    return anchor_examples, handoff_examples, coverage


def write_example_tables(anchor_rows: list[dict], handoff_rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(anchor_rows, output_dir / "anchor_examples.parquet")
    write_table(anchor_rows, output_dir / "anchor_examples.jsonl")
    write_table(handoff_rows, output_dir / "handoff_examples.parquet")
    write_table(handoff_rows, output_dir / "handoff_examples.jsonl")


def rescue_root_counts(rows: Iterable[dict]) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if bool(row.get("complete_pair")):
            grouped[str(row["stat_group_id"])].append(row)
    positive = sum(any(any(int(row.get(f"rescue_{name}", 0)) for name in FINITE_NAMES)
                       for row in group) for group in grouped.values())
    return {"valid_roots": len(grouped), "rescue_positive_roots": positive,
            "rescue_negative_roots": len(grouped) - positive}
