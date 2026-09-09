"""Run sharded stop/continue online episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb3p_stop.io import read_jsonl
from recovery_handback.hb3p_stop.online import OnlineRunner


METHODS = ("NONE", "FIXED_L60", "FIXED_L80", "LEARNED_STOP_CONTINUE")


def _safe(root_id: str) -> str:
    return root_id.replace(":", "__").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not protocol.get("frozen") or not protocol.get("test_locked"):
        raise RuntimeError("stop/continue test execution requires a frozen protocol")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    roots = read_jsonl(args.roots)
    expected = list(map(int, protocol["test_seeds"]))
    if len(roots) != len(expected) or [int(row["seed"]) for row in roots] != expected:
        raise RuntimeError("root manifest does not match frozen test seeds")
    roots = [row for index, row in enumerate(roots) if index % args.num_shards == args.shard_index]
    methods = list(args.methods) if args.methods is not None else ["FIXED_L60", "FIXED_L80", "LEARNED_STOP_CONTINUE"]
    if any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be among {METHODS}")
    runner = OnlineRunner(
        args.config,
        args.protocol,
        device=args.device,
        base_device=args.base_device,
    )
    records = []
    protocol_hash = sha256_file(args.protocol)
    for root in roots:
        for method in methods:
            path = args.output_dir / method / f"{_safe(root['root_id'])}.json"
            if args.resume and path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("engineering_ok") and old.get("protocol_hash") == protocol_hash and Path(old.get("trajectory_path", "")).is_file():
                    records.append(old)
                    continue
            result = runner.run(root, method, path)
            records.append(result)
            print(json.dumps({"root_id": root["root_id"], "method": method, "engineering_ok": result["engineering_ok"], "system_success": result["system_success"], "decision": result.get("stop_continue_decision")}), flush=True)
            if not result["engineering_ok"]:
                raise RuntimeError(f"online episode failed: {root['root_id']} {method}: {result['exception_reason']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({
        "schema_version": "hb3p_stop_continue_online_shard_v1", "protocol_hash": protocol_hash,
        "shard_index": args.shard_index, "num_shards": args.num_shards, "roots": len(roots),
        "methods": methods, "records": len(records), "all_engineering_ok": all(row["engineering_ok"] for row in records),
    }, args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
