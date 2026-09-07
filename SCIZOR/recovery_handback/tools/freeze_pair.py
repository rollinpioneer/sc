"""Freeze the selected HB1 base policy or policy pair without copying weights."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


RUNTIME_SEMANTICS_VERSION = "hb1_runtime_v2_no_extra_observation"


def _evaluation_dirs(path: Path):
    if (path / "summary.json").is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.is_dir() and (item / "summary.json").is_file())


def _select_base(evaluations: Path):
    candidates = []
    for directory in _evaluation_dirs(evaluations):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        checkpoint = str(summary.get("checkpoint", ""))
        match = re.search(r"epoch[_-]?(\d+)", Path(checkpoint).stem)
        training_step = int(match.group(1)) if match else 10**12
        candidates.append((float(summary.get("success_rate", -1.0)), training_step, directory.name, summary))
    if not candidates:
        raise RuntimeError(f"no evaluation summaries found under {evaluations}")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = candidates[0][3]
    checkpoint = Path(chosen["checkpoint"]).resolve()
    return {
        "task": chosen["task"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": chosen.get("checkpoint_sha256") or sha256_file(checkpoint),
        "algorithm": "bc_rnn_gmm", "observation_protocol": "raw_rgb_uint8_hwc_plus_proprio",
        "observation_shapes": chosen.get("observation_shapes", {}),
        "action_dim": int(chosen.get("action_dim", 7)),
        "action_bounds": chosen.get("action_bounds", [[-1.0] * 7, [1.0] * 7]),
        "state_history_protocol": "shadow_update_once_per_env_step",
        "random_tape": "sha256(task,role,root_id,absolute_t,component)",
        "uses_privileged_input": False, "selection_success_rate": candidates[0][0],
        "selected_training_epoch": None if candidates[0][1] == 10**12 else candidates[0][1],
        "selection_evaluations": [
            {"name": name, "success_rate": rate, "training_epoch": None if step == 10**12 else step}
            for rate, step, name, _ in candidates
        ],
        "frozen": True,
    }


def _select_action_policy(summary_path: Path, task: str):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "QUALIFIED":
        raise RuntimeError(
            f"cannot freeze unqualified action policy: {summary.get('status')}"
        )
    selected = summary.get("selected_candidate") or {}
    selection = summary.get("selection") or {}
    checkpoint = Path(selected["checkpoint"]).expanduser().resolve()
    return {
        "schema_version": "hb1_base_policy_v1",
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": selected.get("checkpoint_sha256") or sha256_file(checkpoint),
        "algorithm": "bc_rnn_gmm_visual_distilled",
        "observation_protocol": "raw_rgb_uint8_hwc_plus_proprio",
        "action_dim": int(selection.get("action_dim", 7)),
        "uses_privileged_input": False,
        "selection_success_rate": float(selection.get("success_rate", 0.0)),
        "selected_training_epoch": selected.get("training_steps"),
        "training_sources": [
            "source_success_demonstrations",
            "runtime_privileged_teacher_successes",
        ],
        "random_tape": "sha256(task,role,root_id,absolute_t,component)",
        "state_history_protocol": "shadow_update_once_per_env_step",
        "frozen": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--base-evaluations", type=Path)
    parser.add_argument("--action-policy-evaluation", type=Path)
    parser.add_argument("--base-policy-json", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--base-status", choices=("NEED_BASE_POLICY",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.base_only:
        if args.action_policy_evaluation is not None:
            payload = _select_action_policy(args.action_policy_evaluation, args.task)
            atomic_json_dump(payload, args.output)
            print(json.dumps({
                "task": args.task,
                "checkpoint": payload["checkpoint"],
                "success_rate": payload["selection_success_rate"],
            }, indent=2))
            return
        if args.base_evaluations is None:
            raise SystemExit(
                "--base-evaluations or --action-policy-evaluation is required with --base-only"
            )
        payload = _select_base(args.base_evaluations)
        payload["schema_version"] = "hb1_base_policy_v1"
        atomic_json_dump(payload, args.output)
        print(json.dumps({"task": args.task, "checkpoint": payload["checkpoint"],
                          "success_rate": payload["selection_success_rate"]}, indent=2))
        return
    if args.base_status:
        if args.base_policy_json is None:
            raise SystemExit("--base-policy-json is required with --base-status")
        base = json.loads(args.base_policy_json.read_text(encoding="utf-8"))
        payload = {
            "schema_version": "hb1_policy_pair_status_v1", "task": args.task,
            "runtime_semantics_version": RUNTIME_SEMANTICS_VERSION,
            "status": args.base_status, "base": base, "repair": None,
            "qualification": {
                "status": args.base_status,
                "reason": "all frozen base-policy candidates lacked demonstrated task capability on base_val",
            },
            "frozen": True,
        }
        atomic_json_dump(payload, args.output)
        print(json.dumps({"task": args.task, "status": args.base_status,
                          "output": str(args.output.resolve())}, indent=2))
        return
    if args.base_policy_json is None or args.qualification is None:
        raise SystemExit("policy-pair mode requires --base-policy-json and --qualification")
    base = json.loads(args.base_policy_json.read_text(encoding="utf-8"))
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "hb1_policy_pair_v1", "task": args.task,
        "runtime_semantics_version": RUNTIME_SEMANTICS_VERSION,
        "base": base, "repair": qualification.get("selected_repair", qualification),
        "qualification": qualification, "frozen": True,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"task": args.task, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
