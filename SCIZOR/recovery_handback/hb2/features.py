"""Shared offline/runtime featurizer and feature-cache builder for HB2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, sha256_json

CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _as_rgb(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"expected RGB image, got shape {value.shape}")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = np.transpose(value, (1, 2, 0))
    if value.shape[-1] != 3:
        raise ValueError(f"expected RGB channels, got shape {value.shape}")
    return value.astype(np.uint8, copy=False)


def _proprio(obs: dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for key in PROPRIO_KEYS:
        if key not in obs:
            raise ValueError(f"missing proprio field {key}")
        values.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    value = np.concatenate(values)
    if value.shape != (9,) or not np.isfinite(value).all():
        raise ValueError(f"invalid proprio vector shape/values: {value.shape}")
    return value


def _load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict] | None]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: np.asarray(handle[key]) for key in handle.files}
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    return arrays, meta


def _anchor_history(row: dict, horizon: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    arrays, _ = _load_npz(row["anchor_history_path"])
    anchor_t = int(row["anchor_t"])
    frame_ids = sorted({int(key.split("_")[1]) for key in arrays if key.startswith(f"{anchor_t}_")})
    if not frame_ids:
        raise ValueError(f"empty anchor history: {row['anchor_history_path']}")
    frame_ids = frame_ids[-4:]
    original_count = len(frame_ids)
    original_times = list(range(anchor_t - original_count + 1, anchor_t + 1))
    pad_count = 4 - original_count
    frame_ids = [frame_ids[0]] * pad_count + frame_ids
    absolute_times = [original_times[0]] * pad_count + original_times
    rollout, _ = _load_npz(row["rollout_path"])
    suggestions = np.asarray(rollout["suggestions"], dtype=np.float32)
    frames = []
    for frame_id in frame_ids:
        obs = {key: arrays[f"{anchor_t}_{frame_id}_{key}"] for key in CAMERA_KEYS + PROPRIO_KEYS}
        frames.append(obs)
    times = np.asarray([max(0, value) for value in absolute_times], dtype=np.float32)
    actions = np.stack([suggestions[min(int(t), len(suggestions) - 1)] for t in times], axis=0)
    for index, obs in enumerate(frames):
        if not all(np.isfinite(np.asarray(obs[key])).all() for key in PROPRIO_KEYS):
            raise ValueError(f"non-finite proprio in anchor history {row['anchor_id']}")
    return frames, actions, np.asarray(times / float(horizon), dtype=np.float32)


def _handoff_history(row: dict, horizon: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    arrays, meta = _load_npz(row["handoff_history_path"])
    frame_ids = sorted({int(key.split("_")[0]) for key in arrays if "_" in key and key.split("_", 1)[1] in CAMERA_KEYS})
    if not frame_ids:
        raise ValueError(f"empty handoff history: {row['handoff_history_path']}")
    frame_ids = frame_ids[-4:]
    frame_ids = [frame_ids[0]] * (4 - len(frame_ids)) + frame_ids
    all_meta = meta or []
    times_meta = [int(item["absolute_t"]) for item in all_meta]
    if len(times_meta) < 1:
        raise ValueError(f"handoff history has no absolute_t: {row['handoff_history_path']}")
    frames, actions, times = [], [], []
    for frame_id in frame_ids:
        obs = {key: arrays[f"{frame_id}_{key}"] for key in CAMERA_KEYS + PROPRIO_KEYS}
        frames.append(obs)
        actions.append(np.asarray(arrays[f"{frame_id}_base_action"], dtype=np.float32).reshape(7))
        times.append(times_meta[min(frame_id, len(times_meta) - 1)])
    return frames, np.stack(actions), np.asarray(times, dtype=np.float32) / float(horizon)


class ObservationFeaturizer:
    """The single DINOv2 preprocessing/feature path used offline and online."""

    def __init__(self, config: dict, device: str | None = None):
        self.config = config
        encoder_cfg = config["hb2"]["encoder"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = int(encoder_cfg["input_size"])
        self.mean = torch.tensor(encoder_cfg["mean"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(encoder_cfg["std"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        repo = os.environ.get("HB2_DINO_REPO")
        if repo:
            self.encoder = torch.hub.load(repo, "dinov2_vits14", source="local", pretrained=True)
        else:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        self.encoder.eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder_source = repo or "facebookresearch/dinov2"
        source_candidates = [Path(repo)] if repo else [
            Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        ]
        self.encoder_commit = None
        for source_path in source_candidates:
            if source_path and (source_path / ".git").exists():
                try:
                    self.encoder_commit = subprocess.check_output(
                        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
                        text=True,
                    ).strip()
                except (OSError, subprocess.CalledProcessError):
                    pass

    @torch.inference_mode()
    def encode_images(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(images)
        if array.ndim != 5 or array.shape[2] != 3:
            raise ValueError(f"expected [N,T,C,H,W] image tensor, got {array.shape}")
        tensor = torch.from_numpy(array.astype(np.float32, copy=False) / 255.0).to(self.device)
        n, t, c, h, w = tensor.shape
        tensor = tensor.reshape(n * t, c, h, w)
        tensor = F.interpolate(tensor, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False, antialias=True)
        tensor = (tensor - self.mean) / self.std
        result = self.encoder.forward_features(tensor)
        global_feature = result["x_norm_clstoken"].reshape(n, t, 384).float().cpu().numpy()
        patches = result["x_norm_patchtokens"].reshape(n * t, 16, 16, 384)
        patches = patches.permute(0, 3, 1, 2)
        patches = F.adaptive_avg_pool2d(patches, (4, 4)).permute(0, 2, 3, 1)
        local_feature = patches.reshape(n, t, 16, 384).float().cpu().numpy()
        return global_feature, local_feature

    def featurize_rows(self, rows: list[dict], *, head: str = "anchor", batch_size: int = 64) -> dict[str, Any]:
        globals_, locals_, proprio, actions, times, helper_elapsed = [], [], [], [], [], []
        sample_ids, image_hashes, metadata = [], [], []
        for row in rows:
            if head == "anchor":
                frames, frame_actions, frame_times = _anchor_history(row, int(self.config["horizon_steps"]))
            else:
                if not bool(row.get("eligible")):
                    continue
                frames, frame_actions, frame_times = _handoff_history(row, int(self.config["horizon_steps"]))
            image_array = np.asarray([
                [_as_rgb(frame[key]) for key in CAMERA_KEYS] for frame in frames
            ], dtype=np.uint8)
            # [T, cameras, H, W, C] -> [T, cameras, C, H, W] and flatten cameras as batch.
            image_array = np.transpose(image_array, (0, 1, 4, 2, 3))
            image_hashes.append(hashlib.sha256(image_array.tobytes()).hexdigest())
            flat = image_array.reshape(1, image_array.shape[0] * image_array.shape[1], 3,
                                       image_array.shape[3], image_array.shape[4])
            global_flat, local_flat = self.encode_images(flat)
            globals_.append(global_flat.reshape(4, 2, 384))
            locals_.append(local_flat.reshape(4, 2, 16, 384))
            proprio.append(np.stack([_proprio(frame) for frame in frames]))
            actions.append(frame_actions.astype(np.float32))
            times.append(frame_times.reshape(4, 1))
            elapsed = float(row.get("helper_length", 0)) / float(self.config["horizon_steps"])
            helper_elapsed.append([elapsed])
            sample_id = str(row.get("example_id") or f"F:{row['anchor_id']}")
            sample_ids.append(sample_id)
            metadata.append({
                "example_id": sample_id,
                "root_id": row.get("root_id"),
                "stat_group_id": row.get("stat_group_id"),
                "normalized_time": frame_times.tolist(),
                "absolute_t": np.rint(
                    frame_times * float(self.config["horizon_steps"])
                ).astype(int).tolist(),
                "padding_applied": len(set(frame_times.tolist())) < 4,
            })
        if not sample_ids:
            raise ValueError(f"no feature rows for head={head}")
        return {
            "global": np.asarray(globals_, dtype=np.float16),
            "local": np.asarray(locals_, dtype=np.float16),
            "proprio": np.asarray(proprio, dtype=np.float32),
            "base_actions": np.asarray(actions, dtype=np.float32),
            "time": np.asarray(times, dtype=np.float32),
            "helper_elapsed": np.asarray(helper_elapsed, dtype=np.float32),
            "sample_ids": sample_ids,
            "sample_image_hashes": image_hashes,
            "metadata": metadata,
        }


def _save_cache(cache: dict[str, Any], output_dir: Path, name: str, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{name}.npz", **{key: value for key, value in cache.items()
                                                        if isinstance(value, np.ndarray)})
    atomic_json_dump({"sample_ids": cache["sample_ids"],
                      "sample_image_hashes": cache["sample_image_hashes"],
                      "metadata": cache["metadata"]},
                     output_dir / f"{name}.samples.json")
    atomic_json_dump(manifest, output_dir / f"{name}.manifest.json")


def build_features(config: dict, dataset_dir: Path, roles: list[str], output_dir: Path,
                   *, head: str = "anchor", normalizer: Path | None = None,
                   batch_size: int = 64) -> None:
    table_name = "anchor_examples.parquet" if head == "anchor" else "handoff_examples.parquet"
    rows = read_table(dataset_dir / table_name)
    rows = [row for row in rows if row.get("split") in roles or row.get("data_role") in roles]
    if head == "handoff":
        rows = [row for row in rows if bool(row.get("eligible"))]
    featurizer = ObservationFeaturizer(config)
    cache = featurizer.featurize_rows(rows, head=head, batch_size=batch_size)
    # Normalizers are fit only on the train split and then applied in dataset.py.
    train_rows = [row for row in rows if row.get("split") == "train" or row.get("data_role") in ("legacy_train", "hb2_train", "hb2_train_extra")]
    if train_rows:
        sample_to_index = {sample_id: index for index, sample_id in enumerate(cache["sample_ids"])}
        indices = [sample_to_index[row["example_id"]] for row in train_rows if row["example_id"] in sample_to_index]
        if indices:
            p = cache["proprio"][indices].reshape(-1, 9)
            a = cache["base_actions"][indices].reshape(-1, 7)
            mean = np.concatenate([p.mean(axis=0), a.mean(axis=0)])
            std = np.concatenate([p.std(axis=0), a.std(axis=0)])
            std = np.maximum(std, 1e-6)
            normalizer_payload = {"proprio_mean": mean[:9].tolist(), "proprio_std": std[:9].tolist(),
                                  "action_mean": mean[9:].tolist(), "action_std": std[9:].tolist(),
                                  "fit_split": "train", "head": head}
            atomic_json_dump(normalizer_payload, output_dir / "normalizer.json")
    elif normalizer and Path(normalizer).is_file():
        payload = json.loads(Path(normalizer).read_text(encoding="utf-8"))
        atomic_json_dump(payload, output_dir / "normalizer.json")
    weight_path = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    weights_hash = os.environ.get("HB2_DINO_WEIGHTS_SHA256")
    if not weights_hash and weight_path.is_file():
        weights_hash = sha256_file(weight_path)
    encoder_record = config["hb2"]["encoder"] | {
        "source": featurizer.encoder_source,
        "code_commit": featurizer.encoder_commit,
        "weights_sha256": weights_hash,
        "torch_version": torch.__version__,
    }
    manifest = {
        "schema_version": "hb2_feature_manifest_v1",
        "head": head,
        "roles": roles,
        "sample_count": len(cache["sample_ids"]),
        "encoder": encoder_record,
        "encoder_hash": sha256_json(encoder_record),
        "transform": {"input": "uint8_hwc_rgb", "resize": "224 bicubic antialias", "mean": config["hb2"]["encoder"]["mean"], "std": config["hb2"]["encoder"]["std"]},
        "sample_image_set_hash": sha256_json(cache["sample_image_hashes"]),
        "shapes": {key: list(value.shape) for key, value in cache.items() if isinstance(value, np.ndarray)},
    }
    _save_cache(cache, output_dir, "anchor" if head == "anchor" else "handoff", manifest)
    atomic_json_dump(manifest, output_dir / "feature_manifest.json")
    schema = {
        "schema_version": "hb2_input_schema_v1", "head": head, "cameras": list(CAMERA_KEYS),
        "history_frames": 4, "proprio": list(PROPRIO_KEYS), "base_action_source": "suggestions[t]" if head == "anchor" else "handoff_history.base_action",
        "helper_elapsed": "helper_length / horizon" if head == "handoff" else "not_used",
        "forbidden": ["object", "reward", "success", "future", "root_id", "role", "repair_action"],
    }
    atomic_json_dump(schema, output_dir / "input_schema.json")
    atomic_json_dump({
        "schema_version": "hb2_time_alignment_example_v1",
        "head": head,
        "sample": cache["metadata"][0],
        "sample_image_sha256": cache["sample_image_hashes"][0],
        "tensor_shapes": {
            key: list(value[0].shape) for key, value in cache.items()
            if isinstance(value, np.ndarray)
        },
        "all_inputs_finite": all(
            np.isfinite(value).all() for key, value in cache.items()
            if isinstance(value, np.ndarray)
        ),
    }, output_dir / "time_alignment_example.json")
    print(json.dumps({"head": head, "samples": len(cache["sample_ids"]), "output_dir": str(output_dir)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["train", "validation"])
    parser.add_argument("--head", choices=("anchor", "handoff"), default="anchor")
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args.resume
    build_features(json.loads(args.config.read_text(encoding="utf-8")), args.dataset, args.roles,
                   args.output_dir, head=args.head, normalizer=args.normalizer, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
