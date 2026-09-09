"""Collect the fixed 80 HB3-P baseline roots and a label-free runtime manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, write_table
from recovery_handback.data.collect_roots import collect
from recovery_handback.data.select_anchors import select
from recovery_handback.hb3p.io import mark, write_jsonl


RUNTIME_FIELDS = (
    "schema_version", "task", "role", "root_id", "stat_group_id", "seed",
    "canonical_payload_path", "rollout_path", "anchor_history_path",
    "policy_memory_path", "policy_observation_tape", "initial_state_hash",
    "model_hash", "policy_checkpoint_sha256", "runtime_fingerprint_id",
    "horizon_steps", "control_freq", "exception_reason",
)


def _prior_roots(handback_root: Path) -> list[dict]:
    rows = []
    for pattern in (
        "hb1_v1/roots/**/roots.jsonl",
        "hb1_repair_v1/roots/**/roots.jsonl",
        "hb2_v1/roots/**/roots.jsonl",
    ):
        for path in sorted(handback_root.glob(pattern)):
            rows.extend(read_table(path))
    return rows


def _expected(protocol: dict, start_index: int, count: int) -> list[int]:
    seeds = [int(value) for value in protocol["test_seeds"]]
    return seeds[start_index:start_index + count]


def _valid_shard(path: Path, seeds: list[int]) -> bool:
    roots_path = path / "roots.jsonl"
    if not roots_path.is_file():
        return False
    rows = read_table(roots_path)
    return (
        [int(row["seed"]) for row in rows] == seeds
        and all(row.get("exception_reason") is None for row in rows)
        and all(Path(row["rollout_path"]).is_file() for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not protocol.get("frozen") or not protocol.get("test_locked"):
        raise RuntimeError("test roots may only be collected after final protocol freeze")
    if args.start_index != 0 or args.count != 80:
        raise ValueError("the frozen HB3-P test collection is exactly indices 0..79")
    expected = _expected(protocol, args.start_index, args.count)
    if expected != list(range(500000, 500080)):
        raise RuntimeError("frozen test seeds are not 500000..500079")
    assets = json.loads(Path(config["assets_file"]).read_text(encoding="utf-8"))
    pair_path = Path(config["hb2"]["selected_policy_pair_path"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for first in range(args.start_index, args.start_index + args.count, args.chunk_size):
        size = min(args.chunk_size, args.start_index + args.count - first)
        seeds = expected[first:first + size]
        shard = args.output_dir / "shards" / f"{first:03d}_{first + size - 1:03d}"
        if not (args.resume and _valid_shard(shard, seeds)):
            if shard.exists() and any(shard.iterdir()):
                raise RuntimeError(f"incomplete collection shard requires inspection: {shard}")
            collect(
                config, assets, "square", "hb3p_test", pair_path, shard,
                start_index=first, count=size,
            )
        rows = read_table(shard / "roots.jsonl")
        if [int(row["seed"]) for row in rows] != seeds:
            raise RuntimeError(f"collection shard seed mismatch: {shard}")
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in all_rows] != expected:
        raise RuntimeError("merged test root seed coverage mismatch")
    write_table(all_rows, args.output_dir / "roots.parquet")
    write_table(all_rows, args.output_dir / "roots.jsonl")
    select(config, "square", "hb3p_test", args.output_dir / "roots.parquet", args.output_dir / "anchors.parquet")
    anchors = read_table(args.output_dir / "anchors.parquet")
    roots_by_id = {str(row["root_id"]): row for row in all_rows}
    for row in anchors:
        row["data_role"] = "hb3p_test"
        row["model_hash"] = roots_by_id[str(row["root_id"])]["model_hash"]
    write_table(anchors, args.output_dir / "anchors.parquet")
    write_table(anchors, args.output_dir / "anchors.jsonl")

    prior = _prior_roots(Path(config["output_root"]).parent)
    prior_hashes = {str(row["initial_state_hash"]) for row in prior if row.get("initial_state_hash")}
    new_hashes = [str(row["initial_state_hash"]) for row in all_rows]
    hash_overlap = sorted(prior_hashes.intersection(new_hashes))
    duplicate_hashes = sorted({value for value in new_hashes if new_hashes.count(value) > 1})
    engineering_failures = [
        {"root_id": row["root_id"], "reason": row.get("exception_reason")}
        for row in all_rows if row.get("exception_reason")
    ]
    collection = {
        "schema_version": "hb3p_collected_seed_manifest_v1",
        "protocol_hash": sha256_file(args.protocol),
        "preregistered_seed_manifest_sha256": protocol["test_seed_manifest"]["sha256"],
        "count": len(all_rows),
        "seed_range": [expected[0], expected[-1]],
        "prior_initial_state_hash_overlap": hash_overlap,
        "within_test_duplicate_initial_state_hashes": duplicate_hashes,
        "engineering_failures": engineering_failures,
        "rows": [{
            "root_id": row["root_id"], "stat_group_id": row["stat_group_id"],
            "seed": int(row["seed"]), "initial_state_hash": row["initial_state_hash"],
            "model_hash": row["model_hash"],
        } for row in all_rows],
    }
    atomic_json_dump(collection, args.output_dir / "collected_seed_manifest.json")
    runtime_rows = [{key: row.get(key) for key in RUNTIME_FIELDS} for row in all_rows]
    write_jsonl(runtime_rows, args.output_dir / "runtime_root_manifest.jsonl")
    atomic_json_dump({
        "schema_version": "hb3p_test_roots_summary_v1",
        "protocol_hash": sha256_file(args.protocol),
        "roots": len(all_rows),
        "baseline_successes": sum(bool(row["baseline_success"]) for row in all_rows),
        "anchors": len(anchors),
        "engineering_failures": engineering_failures,
        "initial_state_hash_overlap": hash_overlap,
        "duplicate_initial_state_hashes": duplicate_hashes,
    }, args.output_dir / "summary.json")
    if engineering_failures or hash_overlap or duplicate_hashes:
        raise RuntimeError("HB3-P root collection failed engineering or independence checks")
    mark(Path(config["output_root"]), "hb3p-E-roots.done", "80 frozen test roots collected")
    print(json.dumps({"roots": len(all_rows), "anchors": len(anchors), "baseline_successes": sum(bool(row["baseline_success"]) for row in all_rows)}, indent=2))


if __name__ == "__main__":
    main()
