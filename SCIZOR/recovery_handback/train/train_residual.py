"""Train the fixed HB1 privileged bounded-residual SAC baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from recovery_handback.common import atomic_json_dump
from recovery_handback.train.residual_env import ResidualEnv


class HB1CheckpointCallback(BaseCallback):
    def __init__(self, output_dir, normalizer, checkpoints, metadata):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.normalizer = normalizer
        self.checkpoints = set(int(value) for value in checkpoints)
        self.saved = set()
        self.metadata = dict(metadata)

    def _on_step(self):
        for step in sorted(self.checkpoints):
            if self.num_timesteps >= step and step not in self.saved:
                self.model.save(self.output_dir / f"sac_{step}")
                self.normalizer.save(self.output_dir / f"vecnormalize_{step}.pkl")
                atomic_json_dump({**self.metadata, "sac_transitions": step}, self.output_dir / f"sac_{step}.json")
                self.saved.add(step)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--base-policy-json", type=Path, required=True)
    parser.add_argument("--training-roots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-normalizer", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assets = json.loads(args.assets.read_text())
    base_json = json.loads(args.base_policy_json.read_text())
    checkpoint = base_json.get("checkpoint") or base_json["base"]["checkpoint"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_env = ResidualEnv(
        config, assets, args.task, checkpoint, args.training_roots, device=args.device
    )
    venv = DummyVecEnv([lambda: raw_env])
    if bool(args.resume_checkpoint) != bool(args.resume_normalizer):
        raise SystemExit("--resume-checkpoint and --resume-normalizer must be provided together")
    if args.resume_normalizer:
        venv = VecNormalize.load(str(args.resume_normalizer), venv)
        venv.training = True
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    settings = config["repair_training"]
    metadata = {
        "schema_version": "hb1_repair_checkpoint_v1", "task": args.task,
        "algorithm": "sac_bounded_residual_adapter", "action_mode": "residual",
        "observation_shape": list(raw_env.observation_space.shape),
        "privileged_repairer": True, "base_checkpoint": str(Path(checkpoint).resolve()),
        "total_steps": int(settings["total_steps"]),
        "resumed_from_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "replay_buffer_resumed": False if args.resume_checkpoint else None,
    }
    atomic_json_dump(metadata, args.output_dir / "training_manifest.json")
    callback = HB1CheckpointCallback(args.output_dir, venv, settings["checkpoint_steps"], metadata)
    started = time.time()
    total_transitions = None
    if args.resume_checkpoint:
        model = SAC.load(str(args.resume_checkpoint), env=venv, device=args.device)
        model.set_env(venv)
        already = int(model.num_timesteps)
        remaining = int(settings["total_steps"]) - already
        if remaining <= 0:
            raise RuntimeError(f"resume checkpoint already has {already} transitions")
        callback.saved.update(step for step in callback.checkpoints if step <= already)
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
        total_transitions = int(model.num_timesteps)
    else:
        model = SAC(
            "MlpPolicy", venv, learning_rate=float(settings["learning_rate"]),
            buffer_size=int(settings["buffer_size"]), learning_starts=int(settings["learning_starts"]),
            batch_size=int(settings["batch_size"]), tau=float(settings["tau"]), gamma=float(settings["gamma"]),
            train_freq=int(settings["train_freq"]), gradient_steps=int(settings["gradient_steps"]),
            ent_coef="auto", policy_kwargs={"net_arch": list(settings["net_arch"])},
            seed=int(settings["seed"]), device=args.device, verbose=1,
        )
        model.learn(total_timesteps=int(settings["total_steps"]), callback=callback)
    model.save(args.output_dir / "sac_final")
    model.save_replay_buffer(args.output_dir / "replay_buffer.pkl")
    venv.save(args.output_dir / "vecnormalize_final.pkl")
    atomic_json_dump({
        **metadata, "schema_version": "hb1_repair_training_v1",
        "sac_transitions": total_transitions if total_transitions is not None else raw_env.transition_steps,
        "new_sac_transitions": raw_env.transition_steps,
        "prefix_env_steps": raw_env.prefix_env_steps,
        "base_policy_inference_calls": raw_env.base._total_calls,
        "wall_seconds": time.time() - started, "observation_shape": list(raw_env.observation_space.shape),
    }, args.output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
