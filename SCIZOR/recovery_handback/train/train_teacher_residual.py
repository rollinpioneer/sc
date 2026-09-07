"""Train one fixed phase of teacher-centered residual SAC."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from recovery_handback.common import atomic_json_dump
from recovery_handback.train.teacher_residual_env import TeacherResidualEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("square",), required=True)
    parser.add_argument("--base-policy-json", type=Path, required=True)
    parser.add_argument("--teacher-repair-meta", type=Path, required=True)
    parser.add_argument("--training-roots", type=Path, required=True)
    parser.add_argument("--curriculum-index", type=Path, required=True)
    parser.add_argument("--curriculum-phase", choices=("easy", "medium", "all"), required=True)
    parser.add_argument("--target-total-steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-normalizer", type=Path)
    parser.add_argument("--resume-replay-buffer", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    resume_values = (
        args.resume_checkpoint,
        args.resume_normalizer,
        args.resume_replay_buffer,
    )
    if any(resume_values) and not all(resume_values):
        raise SystemExit("all three resume artifacts must be provided together")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    base = json.loads(args.base_policy_json.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_repair_meta.read_text(encoding="utf-8"))
    base_checkpoint = base.get("checkpoint") or base["base"]["checkpoint"]
    teacher_checkpoint = teacher["checkpoint"]
    settings = config["capability_repair"]["square_teacher_residual"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_env = TeacherResidualEnv(
        config,
        assets,
        args.task,
        base_checkpoint,
        teacher_checkpoint,
        args.training_roots,
        args.curriculum_index,
        args.curriculum_phase,
        device=args.device,
    )
    venv = DummyVecEnv([lambda: raw_env])
    if args.resume_normalizer:
        venv = VecNormalize.load(str(args.resume_normalizer), venv)
        venv.training = True
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    resumed_total = 0
    if args.resume_checkpoint:
        model = SAC.load(str(args.resume_checkpoint), env=venv, device=args.device)
        model.load_replay_buffer(str(args.resume_replay_buffer))
        resumed_total = int(model.num_timesteps)
        prior_summary = args.resume_checkpoint.parent / "training_summary.json"
        if prior_summary.is_file():
            recorded = int(json.loads(prior_summary.read_text(encoding="utf-8"))["total_timesteps"])
            if recorded != resumed_total:
                resumed_total = recorded
                model.num_timesteps = recorded
        remaining = int(args.target_total_steps) - resumed_total
        if remaining <= 0:
            raise RuntimeError(
                f"target {args.target_total_steps} is not above resumed total {resumed_total}"
            )
    else:
        model = SAC(
            "MlpPolicy",
            venv,
            learning_rate=float(settings["learning_rate"]),
            buffer_size=int(settings["buffer_size"]),
            learning_starts=int(settings["learning_starts"]),
            batch_size=int(settings["batch_size"]),
            tau=float(settings["tau"]),
            gamma=float(settings["gamma"]),
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs={"net_arch": list(settings["net_arch"])},
            seed=int(settings["seed"]),
            device=args.device,
            verbose=1,
        )
        remaining = int(args.target_total_steps)

    manifest = {
        "schema_version": "hb1_teacher_residual_training_manifest_v1",
        "task": args.task,
        "curriculum_phase": args.curriculum_phase,
        "target_total_steps": int(args.target_total_steps),
        "resumed_total_steps": resumed_total,
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "resume_normalizer": str(args.resume_normalizer.resolve()) if args.resume_normalizer else None,
        "resume_replay_buffer": str(args.resume_replay_buffer.resolve()) if args.resume_replay_buffer else None,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
        "observation_shape": list(raw_env.observation_space.shape),
        "residual_scale": list(settings["residual_scale"]),
    }
    atomic_json_dump(manifest, args.output_dir / "training_manifest.json")
    started = time.time()
    model.learn(
        total_timesteps=remaining,
        reset_num_timesteps=not bool(args.resume_checkpoint),
    )
    model.save(args.output_dir / "model.zip")
    model.save_replay_buffer(args.output_dir / "replay_buffer.pkl")
    venv.save(args.output_dir / "vecnormalize.pkl")
    summary = {
        **manifest,
        "schema_version": "hb1_teacher_residual_training_v1",
        "algorithm": "sac_teacher_centered_residual",
        "action_mode": "direct",
        "total_timesteps": int(model.num_timesteps),
        "new_sac_transitions": int(raw_env.transition_steps),
        "prefix_env_steps": int(raw_env.prefix_env_steps),
        "base_policy_inference_calls": int(raw_env.base._total_calls),
        "teacher_policy_inference_calls": int(raw_env.teacher._total_calls),
        "replay_buffer_resumed": bool(args.resume_replay_buffer),
        "wall_seconds": time.time() - started,
    }
    atomic_json_dump(summary, args.output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
