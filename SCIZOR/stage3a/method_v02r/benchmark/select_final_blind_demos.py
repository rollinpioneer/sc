"""Select clean, deterministic, previously unseen demos for the final blind test.

The final blind benchmark is deliberately selected from the source datasets at
the last possible moment.  This module only performs the frozen eligibility
operation: numeric demo order, exclusion of every previously used base demo,
two clean replays in the current runtime, and final clean success.  It never
looks at perturbed outcomes or class balance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from stage3a.rescue_v02.common import compare_rollouts, env_for_dataset, replay


def demo_number(name: str) -> int:
    """Return the numeric suffix used by robomimic demo keys."""
    text = str(name)
    try:
        return int(text.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return 10**12


def _json_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _add_identifier(excluded: set[str], value) -> None:
    if value is None:
        return
    text = str(value)
    if text and text not in {"None", "nan", "-1"}:
        excluded.add(text)


def excluded_base_demos(stage1_metadata: Path, v02_bases: Path, confirmation_bases: Path) -> dict[str, set[str]]:
    """Build the task-scoped union of all base demos used before blind test."""
    result: dict[str, set[str]] = {"can": set(), "square": set()}

    for row in _json_rows(stage1_metadata):
        task = str(row.get("task", ""))
        if task not in result:
            continue
        # Stage-1 metadata is normally pair-level, but accepting demo_id and
        # clean_demo_id makes the exclusion robust to older ledgers.
        for key in ("base_demo_id", "demo_id"):
            _add_identifier(result[task], row.get(key))
        clean = row.get("clean_demo_id")
        if clean:
            _add_identifier(result[task], str(clean).removeprefix(f"{task}_").removesuffix("_clean"))

    for path in (v02_bases, confirmation_bases):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = ((str(task), rows) for task, rows in payload.items())
        else:
            items = ((str(row.get("task", "")), [row]) for row in payload)
        for task, rows in items:
            if task not in result:
                continue
            for row in rows or []:
                if isinstance(row, str):
                    _add_identifier(result[task], row)
                elif isinstance(row, dict):
                    _add_identifier(result[task], row.get("demo_id", row.get("base_demo_id")))
    return result


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _check_demo(task: str, source: Path, demo_id: str, handle, env_a, env_b) -> dict:
    group = handle[f"data/{demo_id}"]
    if "states" not in group or "actions" not in group:
        return {
            "task": task,
            "demo_id": demo_id,
            "source_dataset": str(source),
            "eligible": False,
            "determinism_pass": False,
            "final_success": False,
            "reason": "missing_states_or_actions",
        }
    actions = np.asarray(group["actions"][:]).copy()
    if len(actions) == 0 or "states" not in group or len(group["states"]) == 0:
        return {
            "task": task,
            "demo_id": demo_id,
            "source_dataset": str(source),
            "steps": int(len(actions)),
            "eligible": False,
            "determinism_pass": False,
            "final_success": False,
            "reason": "empty_rollout",
        }

    initial = np.asarray(group["states"][0]).copy()
    model_xml = env_a.env.model.get_xml()
    left = replay(env_a, initial, actions, render_images=False, model_xml=model_xml)
    right = replay(env_b, initial, actions, render_images=False, model_xml=model_xml)
    comparison = compare_rollouts(left, right)
    left_success = bool(left["success"][-1]) if len(left["success"]) else False
    right_success = bool(right["success"][-1]) if len(right["success"]) else False
    return {
        "task": task,
        "demo_id": demo_id,
        "source_dataset": str(source),
        "steps": int(len(actions)),
        "eligible": bool(comparison["pass"] and left_success and right_success),
        "determinism_pass": bool(comparison["pass"]),
        "final_success": bool(left_success and right_success),
        "comparison": comparison,
        "reason": "ok" if comparison["pass"] and left_success and right_success else "replay_or_success_failed",
        "selection_rule": "numeric_order_excluded_union_two_clean_replays_final_success",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-source", type=Path, required=True)
    parser.add_argument("--square-source", type=Path, required=True)
    parser.add_argument("--stage1-metadata", type=Path, required=True)
    parser.add_argument("--v02-bases", type=Path, required=True)
    parser.add_argument("--confirmation-bases", type=Path, required=True)
    parser.add_argument("--can-count", type=int, default=10)
    parser.add_argument("--square-count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    if args.can_count < 1 or args.square_count < 1:
        raise ValueError("blind demo counts must be positive")

    excluded = excluded_base_demos(args.stage1_metadata, args.v02_bases, args.confirmation_bases)
    sources = {"can": args.can_source, "square": args.square_source}
    targets = {"can": args.can_count, "square": args.square_count}
    selected: dict[str, list[dict]] = {"can": [], "square": []}
    details: list[dict] = []
    environments = {}
    handles = {}
    try:
        for task in ("can", "square"):
            source = sources[task]
            env_a, _ = env_for_dataset(str(source))
            env_b, _ = env_for_dataset(str(source))
            environments[task] = (env_a, env_b)
            handles[task] = h5py.File(source, "r")
            names = sorted(
                (str(name) for name in handles[task]["data"].keys()),
                key=lambda name: (demo_number(name), name),
            )
            for demo_id in names:
                if len(selected[task]) >= targets[task]:
                    break
                if demo_id in excluded[task]:
                    details.append({
                        "task": task,
                        "demo_id": demo_id,
                        "source_dataset": str(source),
                        "eligible": False,
                        "excluded": True,
                        "reason": "previously_used_base_demo",
                    })
                    continue
                try:
                    row = _check_demo(task, source, demo_id, handles[task], env_a, env_b)
                except Exception as exc:  # keep scanning; the detail is the audit trail
                    row = {
                        "task": task,
                        "demo_id": demo_id,
                        "source_dataset": str(source),
                        "eligible": False,
                        "determinism_pass": False,
                        "final_success": False,
                        "reason": "replay_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                row["excluded"] = False
                # The benchmark generator copies this value into pair
                # metadata.  Mark selected demos explicitly so every
                # downstream blind command sees the frozen ``blind_test``
                # split rather than the generator's pilot default.
                row["split"] = "blind_test"
                details.append(row)
                if row.get("eligible"):
                    selected[task].append(row)
            if len(selected[task]) < targets[task]:
                raise RuntimeError(
                    f"{task}: only {len(selected[task])} eligible demos after exclusions; "
                    f"required {targets[task]}"
                )
    finally:
        for env_a, env_b in environments.values():
            env_a.close()
            env_b.close()
        for handle in handles.values():
            handle.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    # Keep the manifest compatible with generate_benchmark_v02, whose frozen
    # input contract is a task-keyed mapping of demo records.  The audit
    # metadata (counts and rejected candidates) lives in ``--details``.
    payload = {task: rows for task, rows in selected.items()}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.details.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details),
        encoding="utf-8",
    )
    print(json.dumps({"selected": {task: [row["demo_id"] for row in rows] for task, rows in selected.items()}, "tested_or_excluded": len(details)}, indent=2))


if __name__ == "__main__":
    main()
