"""Reconstruct old HB2 validation material as one decision per complete root."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb3p.io import mark, write_jsonl

PROBABILITY_KEYS = (
    "p0", "p_sys_5", "p_sys_20", "p_sys_80",
    "p_genuine_5", "p_genuine_20", "p_genuine_80", "p_full",
)


def _optional_int(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def prepare(config_path: Path, hb2_root: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    roots = read_table(hb2_root / "roots/hb2_val/roots.parquet")
    anchors = read_table(hb2_root / "roots/hb2_val/anchors.parquet")
    anchor_by_id = {str(row["anchor_id"]): row for row in anchors}
    if len(anchor_by_id) != len(anchors):
        raise ValueError("duplicate validation anchor_id")
    branch_rows = []
    for path in sorted((hb2_root / "branches/hb2_val").glob("shard_*/branch_results.parquet")):
        branch_rows.extend(read_table(path))
    branch_index: dict[str, dict[str, dict]] = {}
    seen = set()
    keep_fields = (
        "engineering_ok", "exception_reason", "system_success", "genuine_handoff_success",
        "handoff_success_raw", "success_seen_under_helper", "helper_completed_task",
        "helper_steps_actual", "changed_action_steps", "repair_calls_after_handoff",
        "handoff_executed", "handoff_state_index", "first_raw_success_state",
        "stable_success_state", "episode_end_state_index", "trajectory_path",
    )
    for row in branch_rows:
        key = (str(row["anchor_id"]), str(row["branch_name"]))
        if key in seen:
            raise ValueError(f"duplicate validation branch: {key}")
        seen.add(key)
        anchor = anchor_by_id.get(key[0])
        if anchor is None:
            raise KeyError(f"branch has unknown anchor: {key[0]}")
        root = str(anchor["root_id"])
        branch_index.setdefault(root, {}).setdefault(str(int(anchor["anchor_t"])), {})[key[1]] = {
            field: row.get(field) for field in keep_fields
        }
    expected = len(anchors) * 5
    if len(seen) != expected:
        raise ValueError(f"validation branch coverage mismatch: {len(seen)} != {expected}")

    predictions = {}
    for model in ("M0_time", "M1_proprio", "M4_paired"):
        payload = json.loads((hb2_root / f"metrics/validation/{model}.json").read_text(encoding="utf-8"))
        model_rows = {}
        for row in payload["records"]:
            anchor_id = str(row["example_id"]).removeprefix("F:")
            if anchor_id not in anchor_by_id:
                raise KeyError(f"prediction has unknown validation anchor: {anchor_id}")
            anchor = anchor_by_id[anchor_id]
            key = f"{anchor['root_id']}:{int(anchor['anchor_t'])}"
            if key in model_rows:
                raise ValueError(f"duplicate validation prediction: {model}/{key}")
            model_rows[key] = {name: float(row[name]) for name in PROBABILITY_KEYS}
        if len(model_rows) != len(anchors):
            raise ValueError(f"prediction coverage mismatch for {model}")
        predictions[model] = model_rows

    development_rows = []
    anchors_by_root: dict[str, list[dict]] = {}
    for anchor in anchors:
        anchors_by_root.setdefault(str(anchor["root_id"]), []).append(anchor)
    for root in roots:
        root_id = str(root["root_id"])
        legal = sorted(int(row["anchor_t"]) for row in anchors_by_root.get(root_id, []))
        development_rows.append({
            "root_id": root_id,
            "stat_group_id": str(root.get("stat_group_id") or root_id),
            "seed": int(root["seed"]),
            "canonical_payload_path": str(root["canonical_payload_path"]),
            "rollout_path": str(root["rollout_path"]),
            "policy_memory_path": str(root["policy_memory_path"]),
            "baseline_engineering_ok": root.get("exception_reason") in (None, ""),
            "baseline_exception_reason": root.get("exception_reason"),
            "baseline_system_success": bool(root["baseline_success"]),
            "first_raw_success_state": _optional_int(root.get("first_raw_success_state")),
            "stable_success_state": _optional_int(root.get("stable_success_state")),
            "episode_end_state_index": int(root["actual_steps"]),
            "legal_anchor_times": legal,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(development_rows, output_dir / "development_roots.jsonl")
    atomic_json_dump(branch_index, output_dir / "branch_index.json")
    atomic_json_dump(predictions, output_dir / "cached_predictions.json")
    atomic_json_dump({
        "schema_version": "hb3p_development_manifest_v1",
        "source_split": "hb2_validation_only",
        "roots": len(development_rows),
        "anchors": len(anchors),
        "branches": len(seen),
        "models": sorted(predictions),
        "labels_excluded_from_cached_predictions": True,
        "config_source_ref": config["source_ref"],
    }, output_dir / "manifest.json")
    mark(output_dir.parents[1], "hb3p-B-prepare.done", "validation roots reconstructed")
    return {"roots": len(development_rows), "anchors": len(anchors), "branches": len(seen)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hb2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.hb2_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
