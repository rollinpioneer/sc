"""Build qualification metadata for fixed teacher-centered SAC phases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-meta", type=Path, required=True)
    parser.add_argument("--phase-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = json.loads(args.teacher_meta.read_text(encoding="utf-8"))
    candidates = []
    for phase_dir in args.phase_dirs:
        checkpoint = phase_dir / "model.zip"
        normalizer = phase_dir / "vecnormalize.pkl"
        replay_buffer = phase_dir / "replay_buffer.pkl"
        summary_path = phase_dir / "training_summary.json"
        for path in (checkpoint, normalizer, replay_buffer, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates.append({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "normalizer": str(normalizer.resolve()),
            "normalizer_sha256": sha256_file(normalizer),
            "replay_buffer": str(replay_buffer.resolve()),
            "observation_shape": list(summary["observation_shape"]),
            "algorithm": "sac_teacher_centered_residual",
            "action_mode": "direct",
            "teacher_checkpoint": str(Path(teacher["checkpoint"]).resolve()),
            "teacher_checkpoint_sha256": teacher.get("checkpoint_sha256"),
            "uses_privileged_input": True,
            "training_steps": int(summary["total_timesteps"]),
            "training_unit": "environment_step",
            "curriculum_phase": summary["curriculum_phase"],
            "residual_scale": list(summary["residual_scale"]),
            "action_low": [-1.0] * 7,
            "action_high": [1.0] * 7,
        })
    atomic_json_dump({
        "schema_version": "hb1_teacher_residual_candidates_v1",
        "candidates": candidates,
    }, args.output)
    print(json.dumps({"candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
