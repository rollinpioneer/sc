"""Watch for real BC checkpoints and evaluate each on the fixed base_val seeds."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.evaluation.evaluate_base import evaluate


EPOCH_PATTERN = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)


def _discover(runs: Path, targets: set[int]) -> dict[int, Path]:
    matches: dict[int, list[Path]] = {}
    for checkpoint in runs.rglob("*.pth"):
        match = EPOCH_PATTERN.search(checkpoint.stem)
        if match and int(match.group(1)) in targets:
            matches.setdefault(int(match.group(1)), []).append(checkpoint)
    return {
        epoch: max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        for epoch, paths in matches.items()
    }


def _already_complete(output: Path, checkpoint: Path) -> bool:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rows = summary.get("rows", [])
    return (
        summary.get("checkpoint_sha256") == sha256_file(checkpoint)
        and int(summary.get("roots", 0)) > 0
        and len(rows) == int(summary.get("roots", 0))
        and all(not row.get("exception_reason") and int(row.get("action_dim", 0)) == 7 for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("can", "square"), required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=43200.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    targets = {int(value) for value in config["base_training"]["checkpoint_epochs"]}
    root = Path(config["output_root"])
    started = time.time()
    records: dict[str, dict] = {}
    while True:
        made_progress = False
        for epoch in sorted(targets):
            for task in args.tasks:
                task_run = run_manifest.get("tasks", {}).get(task, {})
                if not task_run.get("run_dir"):
                    raise RuntimeError(f"authoritative run missing for task {task}")
                runs = Path(task_run["run_dir"])
                checkpoint = _discover(runs, targets).get(epoch)
                if checkpoint is None:
                    continue
                output = root / "base" / task / "validation" / f"epoch_{epoch:03d}"
                key = f"{task}:{epoch}"
                if _already_complete(output, checkpoint):
                    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                else:
                    print(json.dumps({"event": "evaluate", "task": task, "epoch": epoch, "checkpoint": str(checkpoint)}, sort_keys=True), flush=True)
                    summary = evaluate(config, assets, task, "base_val", checkpoint, output)
                    made_progress = True
                records[key] = {
                    "task": task, "epoch": epoch, "checkpoint": str(checkpoint.resolve()),
                    "authoritative_run_dir": str(runs.resolve()),
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "roots": summary["roots"], "successes": summary["successes"],
                    "success_rate": summary["success_rate"],
                    "engineering_failures": sum(bool(row.get("exception_reason")) for row in summary.get("rows", [])),
                }
                atomic_json_dump({
                    "schema_version": "hb1_base_checkpoint_candidates_v1",
                    "target_epochs": sorted(targets), "records": sorted(records.values(), key=lambda row: (row["task"], row["epoch"])),
                }, root / "base" / "checkpoint_candidates.json")
        expected = {f"{task}:{epoch}" for task in args.tasks for epoch in targets}
        if expected.issubset(records):
            print(json.dumps({"event": "complete", "evaluations": len(records)}, sort_keys=True), flush=True)
            return
        if time.time() - started >= args.max_wait_seconds:
            missing = sorted(expected - set(records))
            raise TimeoutError(f"timed out waiting for base checkpoints: {missing}")
        if not made_progress:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
