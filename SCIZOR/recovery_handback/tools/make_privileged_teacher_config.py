"""Build the fixed privileged low-dimensional BC-GMM teacher config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from recovery_handback.common import atomic_json_dump


PROPRIO_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def observation_keys(training_view: Path) -> list[str]:
    with h5py.File(training_view, "r") as handle:
        demos = sorted(handle["data"].keys())
        if not demos:
            raise RuntimeError(f"training view has no demos: {training_view}")
        return sorted(handle[f"data/{demos[0]}/obs"].keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    settings_key = "square_teacher" if args.task == "square" else "can_diagnostic"
    settings = protocol["capability_repair"][settings_key]
    keys = observation_keys(args.training_view)
    low_dim_keys = [key for key in PROPRIO_KEYS if key in keys]
    low_dim_keys.extend(
        key for key in keys
        if "object" in key.lower() and key not in low_dim_keys
    )
    if not any("object" in key.lower() for key in low_dim_keys):
        raise RuntimeError(f"no object observation found in {args.training_view}")

    output_root = Path(protocol["output_root"])
    run_root = (
        output_root / "square" / "teacher" / "runs"
        if args.task == "square"
        else output_root / "can" / "diagnostic" / "runs"
    )
    from robomimic.config import config_factory

    cfg = config_factory(algo_name="bc")
    with cfg.unlocked():
        cfg.experiment.name = f"handback_privileged_teacher_{args.task}"
        cfg.experiment.validate = False
        cfg.experiment.rollout.enabled = False
        cfg.experiment.render_video = False
        cfg.experiment.epoch_every_n_steps = settings["steps_per_epoch"]
        cfg.experiment.save.enabled = True
        cfg.experiment.save.every_n_epochs = None
        cfg.experiment.save.epochs = settings["checkpoint_epochs"]
        cfg.experiment.save.on_best_validation = False
        cfg.experiment.save.on_best_rollout_success_rate = False
        cfg.train.data = str(args.training_view.resolve())
        cfg.train.output_dir = str(run_root.resolve())
        cfg.train.num_epochs = settings["epochs"]
        cfg.train.batch_size = settings["batch_size"]
        cfg.train.num_data_workers = 0
        cfg.train.hdf5_cache_mode = "low_dim"
        cfg.train.seq_length = 1
        cfg.train.dataset_keys = ["actions"]
        cfg.train.seed = settings["seed"]
        cfg.train.cuda = True
        cfg.algo.rnn.enabled = False
        cfg.algo.gmm.enabled = True
        cfg.algo.gmm.num_modes = settings.get("gmm_modes", 5)
        cfg.algo.optim_params.policy.learning_rate.initial = settings["learning_rate"]
        cfg.observation.modalities.obs.low_dim = low_dim_keys
        cfg.observation.modalities.obs.rgb = []
        cfg.observation.modalities.obs.depth = []
        cfg.observation.modalities.obs.scan = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(filename=str(args.output.resolve()))
    atomic_json_dump({
        "task": args.task,
        "config": str(args.output.resolve()),
        "training_view": str(args.training_view.resolve()),
        "run_root": str(run_root.resolve()),
        "uses_privileged_input": True,
        "action_mode": "direct",
        "low_dim_keys": low_dim_keys,
        "excluded_inputs": [
            "images", "future_reward", "success_label", "failure_label",
        ],
    }, args.output.with_name("teacher_config_manifest.json"))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
