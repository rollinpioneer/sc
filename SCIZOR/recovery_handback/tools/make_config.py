"""Create the fixed HB1 execution config from a template and assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.template.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    if cfg.get("schema") != "hb1_protocol_v1":
        raise ValueError("unexpected HB1 config schema")
    for task in cfg["tasks"]:
        record = assets.get("tasks", {}).get(task)
        if not isinstance(record, dict):
            raise ValueError(f"missing task asset record: {task}")
        if not record.get("source_hdf5"):
            print(f"WARNING: {task} source HDF5 unresolved; complete HB1-A before runtime.")
    cfg["assets_file"] = str(args.assets.resolve())
    cfg["output_root"] = str(args.output.resolve().parent.parent)
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous != cfg:
            raise ValueError("existing config differs; use an explicit protocol revision")
    atomic_json_dump(cfg, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
