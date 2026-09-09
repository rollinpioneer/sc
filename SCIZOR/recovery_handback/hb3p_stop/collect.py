"""Collect the locked, label-free test roots for the stop/continue pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, write_table
from recovery_handback.data.collect_roots import collect
from recovery_handback.data.select_anchors import select
from recovery_handback.hb3p_stop.io import write_jsonl


def collect_test(config_path: Path, protocol_path: Path, output_dir: Path, *, resume: bool = False) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if not protocol.get("frozen") or not protocol.get("test_locked"):
        raise RuntimeError("test roots require a frozen stop/continue protocol")
    output_dir = Path(output_dir)
    roots_dir = output_dir / "roots"
    roots_dir.mkdir(parents=True, exist_ok=True)
    root_table = roots_dir / "roots.jsonl"
    expected_seeds = list(map(int, protocol["test_seeds"]))
    if not (resume and root_table.is_file()):
        assets = json.loads(Path(config["assets_file"]).read_text(encoding="utf-8"))
        collect_config = dict(config)
        collect_config["roles"] = {config["role"]: {"seed_start": int(config["test_seed_start"]), "roots_per_task": int(config["new_test_roots"])} }
        collect_config["anchor_times"] = [int(config["anchor_t"])]
        collect_config["square_seed_offset"] = 0
        collect(collect_config, assets, "square", config["role"], Path(config["policy_pair_path"]), roots_dir)
    rows = read_table(root_table)
    if [int(row["seed"]) for row in rows] != expected_seeds:
        raise RuntimeError("test root seed coverage does not match frozen protocol")
    anchors = roots_dir / "anchors.parquet"
    if not anchors.is_file():
        select({"anchor_times": [int(config["anchor_t"])], "horizon_steps": 400, "minimum_remaining_steps": 100}, "square", config["role"], roots_dir / "roots.parquet", anchors)
    anchor_rows = read_table(anchors)
    if len(anchor_rows) != len(rows):
        raise RuntimeError(f"expected one t=20 anchor per test root, got {len(anchor_rows)}")
    write_jsonl([{key: row.get(key) for key in ("schema_version", "task", "role", "root_id", "stat_group_id", "seed", "canonical_payload_path", "rollout_path", "anchor_history_path", "policy_memory_path", "initial_state_hash", "model_hash", "policy_checkpoint_sha256", "horizon_steps", "control_freq", "exception_reason")} for row in rows], roots_dir / "runtime_root_manifest.jsonl")
    atomic_json_dump({
        "schema_version": "hb3p_stop_continue_test_roots_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "roots": len(rows),
        "anchors": len(anchor_rows),
        "baseline_successes": sum(bool(row["baseline_success"]) for row in rows),
        "seed_range": [expected_seeds[0], expected_seeds[-1]],
        "label_free": True,
    }, output_dir / "summary.json")
    return {"roots": len(rows), "anchors": len(anchor_rows), "baseline_successes": sum(bool(row["baseline_success"]) for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect_test(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

