"""Cache DINOv2, pre-action state, and action features without outcome labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def valid(path: Path, length: int, state_pad_dim: int) -> bool:
    try:
        with np.load(path) as x:
            return x["image_emb"].shape == (length, 768) and x["states_pre"].shape == (length, state_pad_dim) and x["actions"].shape == (length, 7) and x["state_valid_mask"].shape == (state_pad_dim,)
    except (OSError, KeyError, ValueError):
        return False


def encode(images, processor, encoder, device, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(images), batch_size):
        item = processor(images=list(images[start:start + batch_size]), return_tensors="pt")
        pixels = item["pixel_values"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            z = encoder(pixel_values=pixels).last_hidden_state[:, 0]
            z = torch.nn.functional.normalize(z.float(), dim=-1)
        chunks.append(z.cpu().numpy().astype(np.float16, copy=False))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-index", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--max-groups", type=int)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding="utf-8"))
    labels = pd.read_parquet(a.labels)
    demos = labels[["task", "demo_id", "hdf5_group", "pair_id", "base_demo_id", "variant", "split", "episode_length"]].drop_duplicates("demo_id").sort_values("demo_id")
    if a.max_groups is not None:
        demos = demos.iloc[:a.max_groups].copy()
    with h5py.File(a.benchmark, "r") as h5:
        dims = []
        for row in demos.itertuples(index=False):
            group = h5[f"data/{row.hdf5_group}"]
            if "states_pre" not in group or "actions" not in group or "obs/agentview_image_pre" not in group:
                raise ValueError(f"missing pre-action modality in {row.hdf5_group}")
            if len(group["states_pre"]) != len(group["actions"]) or len(group["obs/agentview_image_pre"]) != len(group["actions"]):
                raise ValueError(f"pre-action alignment mismatch in {row.hdf5_group}")
            dims.append(int(group["states_pre"].shape[1]))
    state_pad_dim = max(dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(config["feature"]["visual_encoder"])
    encoder = AutoModel.from_pretrained(config["feature"]["visual_encoder"]).eval().to(device)
    index_rows = []
    with h5py.File(a.benchmark, "r") as h5:
        for number, row in enumerate(demos.itertuples(index=False), 1):
            group = h5[f"data/{row.hdf5_group}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            raw_state = np.asarray(group["states_pre"], dtype=np.float32)
            images = np.asarray(group["obs/agentview_image_pre"], dtype=np.uint8)
            length, state_dim = raw_state.shape
            if actions.shape != (length, 7) or images.shape[0] != length:
                raise ValueError(f"shape mismatch in {row.hdf5_group}")
            output = a.output_dir / str(row.task) / f"{row.demo_id}.npz"
            if a.overwrite or not valid(output, length, state_pad_dim):
                output.parent.mkdir(parents=True, exist_ok=True)
                padded = np.zeros((length, state_pad_dim), dtype=np.float32)
                padded[:, :state_dim] = raw_state
                mask = np.zeros(state_pad_dim, dtype=np.uint8)
                mask[:state_dim] = 1
                np.savez(output, image_emb=encode(images, processor, encoder, device, 128), states_pre=padded, state_valid_mask=mask, actions=actions)
            index_rows.append({**row._asdict(), "feature_path": str(output), "state_dim": state_dim, "state_pad_dim": state_pad_dim, "image_alignment": "pre_action"})
            print(f"[{number}/{len(demos)}] {row.demo_id}", flush=True)
    index = pd.DataFrame(index_rows).sort_values(["task", "demo_id"]).reset_index(drop=True)
    a.output_index.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(a.output_index, index=False)
    manifest = {
        "visual_encoder": config["feature"]["visual_encoder"], "image_embedding_dim": 768,
        "image_alignment": "pre_action", "state_pad_dim": state_pad_dim, "group_count": int(len(index)),
        "transition_count": int(index.episode_length.sum()), "split_group_counts": {str(k): int(v) for k, v in index.groupby("split").size().items()},
        "split_transition_counts": {str(k): int(v) for k, v in index.groupby("split").episode_length.sum().items()},
        "feature_index_sha256": digest(a.output_index), "labels_sha256": digest(a.labels),
    }
    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    a.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
