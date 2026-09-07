"""Resolve only known HB1 data and model locations.

This resolver is intentionally conservative: placeholder files and known
SCIZOR scoring models are reported, never promoted to action policies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path
from typing import Dict, Iterable, Optional

from recovery_handback.common import atomic_json_dump


TASKS = ("can", "square")
ENV_FILES = (
    "experiments/cr_scizor/stage3a/method_v02r/config/stage3_v02r.env",
    "experiments/cr_scizor/stage3a/replay_rescue_v02_r1/config/source_paths.env",
)


def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    assignment = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = assignment.match(line)
        if not match:
            continue
        key, value = match.groups()
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            values[key] = parsed[0] if parsed else ""
        except ValueError:
            continue
    return values


def _rewrite_path(raw: Optional[str], project_root: Path) -> Optional[Path]:
    if not raw:
        return None
    value = os.path.expanduser(os.path.expandvars(raw))
    direct = Path(value)
    if direct.exists():
        return direct.resolve()
    old_root = "/home/xushijie/work/cr_scizor"
    if value.startswith(old_root + "/"):
        rewritten = project_root / value[len(old_root) + 1 :]
        if rewritten.exists():
            return rewritten.resolve()
    return direct


def _existing_first(values: Iterable[Optional[Path]]) -> Optional[Path]:
    for value in values:
        if value is not None and value.is_file():
            return value
    return None


def _source_metadata(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        env_args = data.attrs.get("env_args", "")
        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8")
        parsed = json.loads(env_args) if isinstance(env_args, str) and env_args else {}
        demos = sorted(data.keys())
        sample = data[demos[0]] if demos else None
        actions_shape = list(sample["actions"].shape) if sample is not None else []
        obs_keys = sorted(sample["obs"].keys()) if sample is not None and "obs" in sample else []
        return {
            "exists": True,
            "bytes": path.stat().st_size,
            "num_demos": len(demos),
            "action_shape_sample": actions_shape,
            "observation_keys_sample": obs_keys,
            "env_name": parsed.get("env_name"),
            "control_freq": parsed.get("env_kwargs", {}).get("control_freq"),
            "camera_names": parsed.get("env_kwargs", {}).get("camera_names", []),
            "env_args": parsed,
        }


def _candidate_models(project_root: Path) -> list[dict]:
    roots = (
        project_root / "experiments/cr_scizor/stage1/baseline",
        project_root / "experiments/cr_scizor/stage2",
        project_root / "experiments/cr_scizor/stage2_iter1",
        project_root / "experiments/cr_scizor/stage3a",
    )
    excluded = ("stage1/baseline", "stage2", "stage2_iter1", "stage3a")
    found: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pt")) + sorted(root.rglob("*.pth")):
            rel = path.relative_to(project_root).as_posix()
            classification = "known_non_action_model" if any(part in rel for part in excluded) else "unclassified"
            found.append({
                "path": str(path.resolve()),
                "classification": classification,
                "reason": "known SCIZOR score/responsibility artifact; not loaded as action policy",
            })
            if len(found) >= 12:
                return found
    return found


def resolve(project_root: Path) -> dict:
    env_values = dict(os.environ)
    for relative in ENV_FILES:
        env_values.update(_read_env_file(project_root / relative))

    defaults = {
        "can": project_root / "data/robomimic/can/ph/image.hdf5",
        "square": project_root / "data/robomimic/square/ph/image.hdf5",
    }
    source_keys = {"can": "CAN_SOURCE", "square": "SQUARE_SOURCE"}
    base_keys = {"can": "BASE_POLICY_CAN", "square": "BASE_POLICY_SQUARE"}
    repair_keys = {"can": "REPAIR_POLICY_CAN", "square": "REPAIR_POLICY_SQUARE"}
    tasks = {}
    missing = []
    for task in TASKS:
        source = _existing_first((_rewrite_path(env_values.get(source_keys[task]), project_root), defaults[task]))
        base = _rewrite_path(env_values.get(base_keys[task]), project_root)
        repair = _rewrite_path(env_values.get(repair_keys[task]), project_root)
        record = {
            "source_hdf5": str(source) if source is not None and source.is_file() else None,
            "base_checkpoint": str(base) if base is not None and base.is_file() else None,
            "repair_checkpoint": str(repair) if repair is not None and repair.is_file() else None,
        }
        if source is None or not source.is_file():
            missing.append(f"{task}.source_hdf5")
            record["source_metadata"] = {"exists": False}
        else:
            record["source_metadata"] = _source_metadata(source)
        if not record["base_checkpoint"]:
            missing.append(f"{task}.base_checkpoint")
        if not record["repair_checkpoint"]:
            missing.append(f"{task}.repair_checkpoint")
        tasks[task] = record
    return {
        "schema": "hb1_assets_v1",
        "project_root": str(project_root.resolve()),
        "tasks": tasks,
        "known_model_candidates": _candidate_models(project_root),
        "missing": missing,
        "notes": [
            "Missing action models trigger HB1-B/C construction; they are not a scientific failure.",
            "Placeholder files and old SCIZOR score/responsibility models are not action policies.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = resolve(args.project_root.resolve())
    atomic_json_dump(payload, args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
