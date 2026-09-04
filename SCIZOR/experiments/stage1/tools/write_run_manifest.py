#!/usr/bin/env python3
"""Write/update a small, portable Stage-1 run manifest."""
import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import yaml


def git_commit(root: Path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gpu_info():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return [line.strip() for line in text.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--command", required=True)
    p.add_argument("--status", default="started")
    p.add_argument("--failed-step")
    p.add_argument("--error-summary")
    args = p.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = Path(__file__).resolve().parents[3]
    data, benchmark = config.get("data", {}), config.get("benchmark", {})
    manifest = {
        "experiment_id": args.experiment_id, "stage": config.get("stage"),
        "substage": config.get("substage"), "benchmark": benchmark.get("name"),
        "task": benchmark.get("task_ids", []), "candidate_pool": data.get("candidate_pool"),
        "target_demo_ids": data.get("target_demo_ids", []),
        "selection_budget": config.get("selection", {}).get("budget"),
        "data_seed": data.get("data_seed"), "train_seed": config.get("train", {}).get("train_seed"),
        "eval_seed": config.get("eval", {}).get("eval_seed"),
        "policy_backbone": config.get("policy", {}).get("backbone"),
        "influence_method": config.get("influence", {}).get("method"),
        "config_path": str(config_path), "git_commit": git_commit(root), "command": args.command,
        "start_time": dt.datetime.now(dt.timezone.utc).isoformat(), "status": args.status,
        "hostname": socket.gethostname(), "python": sys.version, "gpus": gpu_info(),
    }
    if args.failed_step:
        manifest["failed_step"] = args.failed_step
    if args.error_summary:
        manifest["error_summary"] = args.error_summary
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
