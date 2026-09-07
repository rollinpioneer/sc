"""Build the fixed Robomimic BC-RNN-GMM configuration for one task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.training_view.parent / "bc_source_manifest.json").read_text(encoding="utf-8"))
    if manifest["task"] != args.task:
        raise ValueError("training view task does not match requested task")
    from robomimic.config import config_factory

    cfg = config_factory(algo_name="bc")
    with cfg.unlocked():
        cfg.experiment.name = f"handback_base_{args.task}"
        cfg.experiment.validate = False
        cfg.experiment.rollout.enabled = False
        cfg.experiment.render_video = False
        cfg.experiment.epoch_every_n_steps = protocol["base_training"]["steps_per_epoch"]
        cfg.experiment.save.enabled = True
        cfg.experiment.save.every_n_epochs = None
        cfg.experiment.save.epochs = protocol["base_training"]["checkpoint_epochs"]
        cfg.experiment.save.on_best_rollout_success_rate = False
        cfg.experiment.save.on_best_validation = False
        cfg.train.data = str(args.training_view.resolve())
        cfg.train.output_dir = str((Path(protocol["output_root"]) / "base" / args.task / "runs").resolve())
        cfg.train.num_epochs = protocol["base_training"]["epochs"]
        cfg.train.batch_size = protocol["base_training"]["batch_size"]
        cfg.train.num_data_workers = 2
        cfg.train.hdf5_cache_mode = "low_dim"
        cfg.train.seq_length = protocol["base_training"]["seq_length"]
        cfg.train.dataset_keys = ["actions"]
        cfg.train.seed = protocol["base_training"]["seed"]
        cfg.train.cuda = True
        cfg.algo.rnn.enabled = True
        cfg.algo.rnn.open_loop = False
        cfg.algo.rnn.horizon = protocol["base_training"]["seq_length"]
        cfg.algo.rnn.hidden_dim = 400
        cfg.algo.rnn.num_layers = 2
        cfg.algo.gmm.enabled = True
        cfg.algo.gmm.num_modes = 5
        cfg.algo.actor_layer_dims = []
        cfg.algo.optim_params.policy.learning_rate.initial = protocol["base_training"]["learning_rate"]
        cfg.observation.modalities.obs.low_dim = manifest["proprio_keys"]
        cfg.observation.modalities.obs.rgb = manifest["camera_keys"]
        cfg.observation.modalities.obs.depth = []
        cfg.observation.modalities.obs.scan = []
    cfg.dump(filename=str(args.output.resolve()))
    atomic_json_dump({
        "task": args.task,
        "config": str(args.output.resolve()),
        "training_view": str(args.training_view.resolve()),
        "camera_keys": manifest["camera_keys"],
        "proprio_keys": manifest["proprio_keys"],
        "policy_inputs_excluded": manifest["excluded_from_policy_input"],
    }, args.output.with_name("base_config_manifest.json").resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
