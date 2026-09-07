"""Build an external-link training view from source and successful teacher data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from recovery_handback.common import atomic_json_dump
from recovery_handback.data.collect_teacher_rollouts import STUDENT_OBS_KEYS


def _transition_count(handle: h5py.File, demo_ids: list[str]) -> int:
    return sum(int(handle[f"data/{demo_id}"].attrs["num_samples"]) for demo_id in demo_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--teacher-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    with h5py.File(args.source_view, "r") as source, h5py.File(args.teacher_rollouts, "r") as teacher:
        source_ids = sorted(source["data"].keys())
        teacher_ids = sorted(teacher["data"].keys())
        if set(source_ids) & set(teacher_ids):
            raise RuntimeError("source and teacher demo IDs overlap")
        source_transitions = _transition_count(source, source_ids)
        teacher_transitions = _transition_count(teacher, teacher_ids)
        with h5py.File(args.output, "w") as output:
            data = output.create_group("data")
            for key, value in source["data"].attrs.items():
                data.attrs[key] = value
            for demo_id in source_ids:
                data[demo_id] = h5py.ExternalLink(
                    str(args.source_view.resolve()), f"/data/{demo_id}"
                )
            for demo_id in teacher_ids:
                if not demo_id.startswith("runtime_teacher_"):
                    raise RuntimeError(f"unexpected teacher demo ID: {demo_id}")
                data[demo_id] = h5py.ExternalLink(
                    str(args.teacher_rollouts.resolve()), f"/data/{demo_id}"
                )
            data.attrs["total"] = source_transitions + teacher_transitions
            data.attrs["hb1_distilled_view"] = True
    payload = {
        "schema_version": "hb1_distilled_training_view_v1",
        "task": "can",
        "training_view": str(args.output.resolve()),
        "source_view": str(args.source_view.resolve()),
        "teacher_rollouts": str(args.teacher_rollouts.resolve()),
        "source_trajectories": len(source_ids),
        "source_transitions": source_transitions,
        "runtime_teacher_trajectories": len(teacher_ids),
        "runtime_teacher_transitions": teacher_transitions,
        "camera_keys": [key for key in STUDENT_OBS_KEYS if key.endswith("_image")],
        "proprio_keys": [key for key in STUDENT_OBS_KEYS if not key.endswith("_image")],
        "student_observation_keys": list(STUDENT_OBS_KEYS),
        "excluded_student_inputs": [
            "object", "future_reward", "success_label", "failure_label",
        ],
        "failed_teacher_rollouts_in_supervision": 0,
    }
    atomic_json_dump(payload, args.manifest)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
