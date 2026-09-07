"""Build the fixed distilled Can BC-RNN-GMM visual student config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=("can",), required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["task"] != args.task:
        raise ValueError("student manifest task mismatch")
    if "object" not in manifest["excluded_student_inputs"]:
        raise ValueError("student manifest must explicitly exclude object")
    settings = protocol["capability_repair"]["can_visual"]
    run_root = Path(protocol["output_root"]) / "can" / "visual" / "runs"
    from robomimic.config import config_factory

    cfg = config_factory(algo_name="bc")
    with cfg.unlocked():
        cfg.experiment.name = "handback_visual_student_can"
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
        cfg.train.num_data_workers = 2
        # Cache the joint low-dimensional and RGB view once. The distilled
        # view is intentionally external-linked, so per-batch image reads
        # otherwise dominate this fixed visual training run.
        cfg.train.hdf5_cache_mode = "all"
        cfg.train.seq_length = settings["seq_length"]
        cfg.train.dataset_keys = ["actions"]
        cfg.train.seed = settings["seed"]
        cfg.train.cuda = True
        cfg.algo.rnn.enabled = True
        cfg.algo.rnn.open_loop = False
        cfg.algo.rnn.horizon = settings["seq_length"]
        cfg.algo.rnn.hidden_dim = 400
        cfg.algo.rnn.num_layers = 2
        cfg.algo.gmm.enabled = True
        cfg.algo.gmm.num_modes = 5
        cfg.algo.actor_layer_dims = []
        cfg.algo.optim_params.policy.learning_rate.initial = settings["learning_rate"]
        cfg.observation.modalities.obs.low_dim = manifest["proprio_keys"]
        cfg.observation.modalities.obs.rgb = manifest["camera_keys"]
        cfg.observation.modalities.obs.depth = []
        cfg.observation.modalities.obs.scan = []
        cfg.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
        cfg.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = settings["crop_height"]
        cfg.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = settings["crop_width"]
        cfg.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
        cfg.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(filename=str(args.output.resolve()))
    atomic_json_dump({
        "schema_version": "hb1_visual_student_config_manifest_v1",
        "task": args.task,
        "config": str(args.output.resolve()),
        "training_view": str(args.training_view.resolve()),
        "run_root": str(run_root.resolve()),
        "camera_keys": manifest["camera_keys"],
        "proprio_keys": manifest["proprio_keys"],
        "excluded_inputs": manifest["excluded_student_inputs"],
        "uses_privileged_input": False,
        "training_sources": [
            "source_success_demonstrations",
            "runtime_privileged_teacher_successes",
        ],
    }, args.output.with_name("student_config_manifest.json"))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
