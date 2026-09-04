#!/usr/bin/env python3
"""Fail clearly before a Stage-1 run if reproducibility inputs are absent."""
import argparse
import sys
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    path = Path(args.config)
    c = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errors = []
    data = c.get("data", {})
    root = Path(str(data.get("root", "")))
    if not root.is_dir(): errors.append(f"data.root does not exist: {root}")
    if not data.get("target_demo_ids"): errors.append("target_demo_ids are not frozen from a real dataset")
    pool = Path(str(data.get("candidate_pool", "")))
    if not pool.is_file() or not [x for x in pool.read_text(encoding="utf-8").splitlines() if x and not x.startswith("#")]:
        errors.append(f"candidate pool is not frozen: {pool}")
    if c.get("selection", {}).get("budget") is None: errors.append("selection.budget is not frozen")
    if not c.get("pipeline", {}).get("command"): errors.append("original DataMIL pipeline.command is unavailable")
    if errors:
        print("STAGE1_PREFLIGHT_BLOCKED", file=sys.stderr)
        print("\n".join(f"- {e}" for e in errors), file=sys.stderr)
        raise SystemExit(2)
    print("STAGE1_PREFLIGHT_OK")


if __name__ == "__main__": main()
