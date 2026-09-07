"""Run all paired HB1 branches for a deterministic anchor shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_json, stable_seed, write_table
from recovery_handback.execution.label_reference import BRANCH_SPECS
from recovery_handback.execution.paired_branch import run_branch


def _slug(anchor_id: str) -> str:
    return anchor_id.replace(":", "__").replace("/", "_")


def _complete(path: Path, config_hash: str, pair_hash: str) -> bool:
    if not path.is_file() or not path.with_suffix(".npz").is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(row.get("engineering_ok"))
        and row.get("config_hash") == config_hash
        and row.get("policy_pair_hash") == pair_hash
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-pair", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--role", default="probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-device")
    parser.add_argument("--repair-device")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard selection")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pair = json.loads(args.policy_pair.read_text(encoding="utf-8"))
    config_hash = sha256_json(config)
    pair_hash = sha256_json(pair)
    anchors = [
        row for row in read_table(args.anchors)
        if row.get("role") == args.role and stable_seed(row["anchor_id"]) % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for anchor in anchors:
        anchor_dir = args.output_dir / "records" / _slug(anchor["anchor_id"])
        for branch_name, length in BRANCH_SPECS:
            output = anchor_dir / f"{branch_name}.json"
            if args.resume and _complete(output, config_hash, pair_hash):
                result = json.loads(output.read_text(encoding="utf-8"))
            else:
                result = run_branch(
                    config, anchor, pair, length, output, device=args.device,
                    base_device=args.base_device, repair_device=args.repair_device,
                )
            results.append(result)
            print(json.dumps({
                "anchor_id": anchor["anchor_id"], "branch": branch_name,
                "engineering_ok": result["engineering_ok"], "system_success": result["system_success"],
                "genuine_handoff_success": result["genuine_handoff_success"],
            }, sort_keys=True), flush=True)
    write_table(results, args.output_dir / f"branch_results_shard_{args.shard_index:03d}.parquet")
    all_rows = []
    for path in sorted((args.output_dir / "records").glob("*/*.json")):
        try:
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    write_table(all_rows, args.output_dir / "branch_results.parquet")
    atomic_json_dump({
        "schema_version": "hb1_branch_run_v1", "role": args.role,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "anchors_in_shard": len(anchors), "branches_written": len(results),
        "config_hash": config_hash, "policy_pair_hash": pair_hash,
    }, args.output_dir / f"manifest_shard_{args.shard_index:03d}.json")


if __name__ == "__main__":
    main()
