"""Run sharded complete HB3-P online episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb3p.io import read_jsonl
from recovery_handback.hb3p.online_episode import OnlineRunner


def _safe(root_id: str) -> str:
    return root_id.replace(":", "__").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not protocol.get("frozen") or not protocol.get("test_locked"):
        raise RuntimeError("formal online execution requires the frozen test protocol")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    protocol_hash = sha256_file(args.protocol)
    all_roots = read_jsonl(args.roots)
    root_ids = [str(row["root_id"]) for row in all_roots]
    seeds = [int(row["seed"]) for row in all_roots]
    if len(all_roots) != 80 or len(set(root_ids)) != 80:
        raise RuntimeError("formal online execution requires 80 unique preregistered roots")
    if sorted(seeds) != list(range(500000, 500080)):
        raise RuntimeError("formal online root seeds do not match the frozen 500000..500079 manifest")
    roots = all_roots
    roots = [row for index, row in enumerate(roots) if index % args.num_shards == args.shard_index]
    alias_map = {row["alias"]: row["canonical_execution"] for row in protocol.get("method_aliases", [])}
    explicitly_requested = args.methods is not None
    requested = args.methods if explicitly_requested else list(protocol["methods"])
    methods = []
    for method in requested:
        if method not in protocol["methods"]:
            raise ValueError(f"method is not in the frozen protocol: {method}")
        canonical = alias_map.get(method, method)
        if canonical == "NONE" and not explicitly_requested:
            continue
        if canonical not in methods:
            methods.append(canonical)
    runner = OnlineRunner(args.config, args.protocol, args.queue_dir)
    records = []
    for root in roots:
        for method in methods:
            output = args.output_dir / method / f"{_safe(root['root_id'])}.json"
            if args.resume and output.is_file():
                old = json.loads(output.read_text(encoding="utf-8"))
                trajectory = Path(old.get("trajectory_path", ""))
                reusable = (
                    old.get("protocol_hash") == protocol_hash
                    and old.get("semantic_pair_id") == protocol["semantic_pair_id"]
                    and old.get("method_id") == method
                    and old.get("root_id") == root["root_id"]
                    and int(old.get("root_seed", -1)) == int(root["seed"])
                    and old.get("initial_state_hash") == root.get("initial_state_hash")
                    and old.get("engineering_ok")
                    and trajectory.is_file()
                )
                if reusable:
                    records.append(old)
                    continue
            result = runner.run(root, method, output)
            records.append(result)
            print(json.dumps({"root_id": root["root_id"], "method": method, "engineering_ok": result["engineering_ok"], "system_success": result["system_success"]}), flush=True)
            if not result["engineering_ok"]:
                raise RuntimeError(f"online episode failed: {root['root_id']} {method}: {result['exception_reason']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({
        "schema_version": "hb3p_online_shard_summary_v1",
        "protocol_hash": protocol_hash,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "roots": len(roots),
        "methods": methods,
        "records": len(records),
        "all_engineering_ok": all(row["engineering_ok"] for row in records),
    }, args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
