"""Create a deterministic, external-link training view of a source HDF5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from recovery_handback.common import atomic_json_dump


def split_key(task: str, demo_id: str, seed: int) -> str:
    text = f"{task}\x1f{demo_id}\x1f{seed}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def build_view(source: Path, output_dir: Path, task: str, split_seed: int, fraction: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_view = output_dir / "source_train.hdf5"
    manifest_path = output_dir / "bc_source_manifest.json"
    if train_view.exists():
        train_view.unlink()
    with h5py.File(source, "r") as src:
        data = src["data"]
        demo_ids = sorted(data.keys())
        ordered = sorted(demo_ids, key=lambda demo: split_key(task, demo, split_seed))
        n_train = int(len(ordered) * fraction)
        train_ids = ordered[:n_train]
        heldout_ids = ordered[n_train:]
        first = data[train_ids[0]] if train_ids else data[ordered[0]]
        obs_keys = sorted(first["obs"].keys())
        camera_keys = [key for key in obs_keys if key.endswith("_image")]
        proprio_keys = [
            key
            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
            if key in obs_keys
        ]
        with h5py.File(train_view, "w") as dst:
            dst_data = dst.create_group("data")
            for key, value in data.attrs.items():
                dst_data.attrs[key] = value
            for demo_id in train_ids:
                dst_data[demo_id] = h5py.ExternalLink(str(source.resolve()), f"/data/{demo_id}")
            dst_data.attrs["hb1_source_hdf5"] = str(source.resolve())
            dst_data.attrs["hb1_task"] = task
            dst_data.attrs["hb1_split_seed"] = int(split_seed)
            dst_data.attrs["hb1_train_fraction"] = float(fraction)
    manifest = {
        "schema": "hb1_bc_source_manifest_v1",
        "task": task,
        "source_hdf5": str(source.resolve()),
        "training_view": str(train_view.resolve()),
        "split_seed": int(split_seed),
        "train_fraction": float(fraction),
        "all_demo_count": len(ordered),
        "train_demo_ids": train_ids,
        "heldout_demo_ids": heldout_ids,
        "camera_keys": camera_keys,
        "proprio_keys": proprio_keys,
        "excluded_from_policy_input": ["object", "reward", "success", "failure_label"],
    }
    atomic_json_dump(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    source = Path(assets["tasks"][args.task]["source_hdf5"])
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest = build_view(
        source=source,
        output_dir=args.output_dir,
        task=args.task,
        split_seed=int(config["source_demo_split_seed"]),
        fraction=float(config["source_train_fraction"]),
    )
    print(json.dumps({k: manifest[k] for k in ("task", "training_view", "all_demo_count", "camera_keys", "proprio_keys")}, indent=2))


if __name__ == "__main__":
    main()
