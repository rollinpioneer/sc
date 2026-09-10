"""HB4 fixed-budget BC-RNN-GMM fine-tuning from the frozen base checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery_handback.common import atomic_json_dump, sha256_file


BASE = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")
BASE_MANIFEST = Path("/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb1_v1/base/square/data/bc_source_manifest.json")
EXPORT = Path("/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1/experiments/handback/hb4_square_absorb_v1")
RUN = Path("/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1")
ARMS = ("REPLAY_ONLY", "MATCHED_STANDARD_DATA", "FIXED_L80_RECOVERY", "HANDOFF_RECOVERY")


def _combine(a: dict, b: dict) -> dict:
    result = {}
    for key in a:
        if isinstance(a[key], dict):
            result[key] = _combine(a[key], b[key])
        else:
            result[key] = torch.cat((a[key], b[key]), dim=0)
    return result


def _state_digest(state: dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        value = state[key]
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()


def _checkpoint_payload(model, config, ckpt, shape_meta, step: int, seed: int, arm: str) -> dict:
    return {
        "algo_name": ckpt["algo_name"], "config": ckpt["config"],
        "shape_metadata": ckpt["shape_metadata"], "env_metadata": ckpt["env_metadata"],
        "obs_normalization_stats": ckpt.get("obs_normalization_stats"),
        "model": {key: value.detach().cpu() for key, value in model.serialize().items()},
        "hb4_metadata": {
            "arm": arm, "training_seed": seed, "optimizer_updates": step,
            "base_checkpoint_sha256": sha256_file(BASE), "strict_base_load": True,
            "learning_rate": 3e-5, "batch_sequences": 32, "sequence_length": 10,
            "source_mixture": [0.5, 0.5], "state_digest": _state_digest(model.serialize()),
        },
    }


def train(arm: str, seed: int, output_dir: Path, device: str, updates: int = 4000) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    from robomimic.algo import algo_factory
    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import obs_utils as ObsUtils
    from robomimic.utils.dataset import SequenceDataset

    with BASE_MANIFEST.open(encoding="utf-8") as handle:
        base_manifest = json.load(handle)
    hdf5_manifest = json.loads((EXPORT / "data/training_hdf5_manifest.json").read_text(encoding="utf-8"))
    arm_hdf5 = Path(hdf5_manifest["arms"][arm])
    if not arm_hdf5.is_file():
        raise FileNotFoundError(arm_hdf5)
    # Keep checkpoint deserialization on CPU; the model is moved to the
    # selected CUDA device only after its structure has been built.
    base_ckpt = torch.load(str(BASE), map_location="cpu", weights_only=False)
    config, ckpt = FileUtils.config_from_checkpoint(ckpt_dict=base_ckpt, verbose=False)
    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = ckpt["shape_metadata"]
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = algo_factory(config.algo_name, config, shape_meta["all_shapes"], shape_meta["ac_dim"], torch_device)
    model_keys = set(model.serialize())
    checkpoint_keys = set(ckpt["model"])
    if model_keys != checkpoint_keys:
        raise RuntimeError(f"strict key mismatch: missing={sorted(model_keys-checkpoint_keys)}, unexpected={sorted(checkpoint_keys-model_keys)}")
    model.deserialize(ckpt["model"])
    model.set_train()
    model.optimizers["policy"] = torch.optim.Adam(model.nets["policy"].parameters(), lr=3e-5)
    model.lr_schedulers["policy"] = None
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_keys = list(shape_meta["all_obs_keys"])
    d0 = SequenceDataset(
        hdf5_path=base_manifest["training_view"], obs_keys=obs_keys, dataset_keys=["actions"],
        seq_length=10, pad_frame_stack=False, pad_seq_length=False, hdf5_cache_mode="low_dim",
        load_next_obs=False,
    )
    new_filter = None if arm == "REPLAY_ONLY" else "new"
    added = SequenceDataset(
        hdf5_path=str(arm_hdf5), obs_keys=obs_keys, dataset_keys=["actions"],
        seq_length=10, pad_frame_stack=False, pad_seq_length=False, hdf5_cache_mode="low_dim",
        load_next_obs=False, filter_by_attribute=new_filter,
    )
    if len(d0) < 16 or len(added) < 16:
        raise RuntimeError(f"insufficient training sequences: d0={len(d0)} added={len(added)}")
    old_loader = DataLoader(d0, batch_size=16, shuffle=True, drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(seed + 1000))
    new_loader = DataLoader(added, batch_size=16, shuffle=True, drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(seed + 2000))
    old_iter, new_iter = iter(old_loader), iter(new_loader)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_digest = _state_digest(model.serialize())
    torch.save(_checkpoint_payload(model, config, ckpt, shape_meta, 0, seed, arm), output_dir / "model_step_0000.pth")
    losses = []
    started = time.time()
    for step in range(1, updates + 1):
        try:
            old_batch = next(old_iter)
        except StopIteration:
            old_iter = iter(old_loader)
            old_batch = next(old_iter)
        try:
            new_batch = next(new_iter)
        except StopIteration:
            new_iter = iter(new_loader)
            new_batch = next(new_iter)
        batch = _combine(old_batch, new_batch)
        processed = model.process_batch_for_training(batch)
        info = model.train_on_batch(processed, step, validate=False)
        loss = float(info["losses"]["action_loss"].detach().cpu())
        losses.append(loss)
        if step in (1000, 2000, updates):
            torch.save(_checkpoint_payload(model, config, ckpt, shape_meta, step, seed, arm), output_dir / f"model_step_{step:04d}.pth")
            atomic_json_dump({"arm": arm, "training_seed": seed, "optimizer_updates": step, "loss": loss, "state_digest": _state_digest(model.serialize())}, output_dir / f"checkpoint_step_{step:04d}.json")
    summary = {
        "schema_version": "hb4_training_summary_v1", "status": "PASS_HB4_E_TRAINED",
        "arm": arm, "training_seed": seed, "optimizer_updates": updates,
        "base_sequences_per_batch": 16, "added_sequences_per_batch": 16,
        "sequence_length": 10, "expected_added_action_exposures": updates * 16 * 10,
        "base_checkpoint": str(BASE.resolve()), "base_checkpoint_sha256": sha256_file(BASE),
        "initial_state_digest": initial_digest, "final_state_digest": _state_digest(model.serialize()),
        "checkpoint_4000": str((output_dir / f"model_step_{updates:04d}.pth").resolve()),
        "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
        "wall_seconds": time.time() - started, "device": str(torch_device),
    }
    atomic_json_dump(summary, output_dir / "training_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--updates", type=int, default=4000)
    args = parser.parse_args()
    print(json.dumps(train(args.arm, args.seed, args.output_dir, args.device, args.updates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
