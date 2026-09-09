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
    protocol_hash = sha256_file(args.protocol)
    roots = read_jsonl(args.roots)
    roots = [row for index, row in enumerate(roots) if index % args.num_shards == args.shard_index]
    alias_map = {row["alias"]: row["canonical_execution"] for row in protocol.get("method_aliases", [])}
    requested = args.methods or list(protocol["methods"])
    methods = []
    for method in requested:
        canonical = alias_map.get(method, method)
        if canonical != "NONE" and canonical not in methods:
            methods.append(canonical)
    runner = OnlineRunner(args.config, args.protocol, args.queue_dir)
    records = []
    for root in roots:
        for method in methods:
            output = args.output_dir / method / f"{_safe(root['root_id'])}.json"
            if args.resume and output.is_file():
                old = json.loads(output.read_text(encoding="utf-8"))
                trajectory = Path(old.get("trajectory_path", ""))
                if old.get("protocol_hash") == protocol_hash and old.get("engineering_ok") and trajectory.is_file():
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
