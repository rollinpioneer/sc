"""Extract frozen DINOv2, state, and action features from the Stage 1 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_INPUT_DATASETS = {"actions", "states", "obs/agentview_image"}
FORBIDDEN_MODEL_INPUT_NAMES = {
    "rewards", "staged_rewards", "success", "failure_onset", "failure_type",
    "responsible_t", "intervention_t", "perturbation_type", "magnitude",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", type=Path, required=True)
    parser.add_argument("--pair-metadata", type=Path, required=True)
    parser.add_argument("--transition-labels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-demos", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_identity(metadata_path: Path, labels_path: Path) -> tuple[dict[str, str], dict[tuple[str, str], str], pd.DataFrame]:
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
    pair_to_demo = {str(row["pair_id"]): str(row["perturbed_demo_id"]) for row in metadata}
    clean_key_to_demo = {(str(row["task"]), str(row["base_demo_id"])): str(row["clean_demo_id"]) for row in metadata}
    labels = pd.read_parquet(labels_path)
    columns = ["task", "demo_id", "split"]
    demos = labels[columns].drop_duplicates(["task", "demo_id"])
    return pair_to_demo, clean_key_to_demo, demos.set_index(["task", "demo_id"])


def valid_existing(path: Path, episode_length: int, state_pad_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as item:
            return (
                item["image_emb"].shape == (episode_length, 768)
                and item["states"].shape == (episode_length, state_pad_dim)
                and item["actions"].shape == (episode_length, 7)
                and item["state_valid_mask"].shape == (state_pad_dim,)
            )
    except (OSError, KeyError, ValueError):
        return False


def resolve_identity(attrs: Any, pair_to_demo: dict[str, str], clean_key_to_demo: dict[tuple[str, str], str]) -> tuple[str, str, str | None, str, str]:
    def attr(name: str) -> Any:
        value = attrs[name]
        return value.decode() if isinstance(value, bytes) else value

    task, variant, base_demo_id = str(attr("task")), str(attr("variant")), str(attr("base_demo_id"))
    pair_id = str(attr("pair_id")) if variant == "perturbed" else None
    if variant == "perturbed":
        demo_id = pair_to_demo[pair_id]
    elif variant == "clean":
        demo_id = clean_key_to_demo[(task, base_demo_id)]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return task, demo_id, pair_id, variant, base_demo_id


def encode_images(images: np.ndarray, processor: Any, encoder: Any, device: torch.device, batch_size: int) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        inputs = processor(images=list(images[start : start + batch_size]), return_tensors="pt")
        pixels = inputs["pixel_values"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embedding = encoder(pixel_values=pixels).last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        output.append(embedding.cpu().numpy().astype(np.float16, copy=False))
    return np.concatenate(output, axis=0)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pair_to_demo, clean_key_to_demo, demo_splits = load_identity(args.pair_metadata, args.transition_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.benchmark_hdf5, "r") as h5:
        group_names = sorted(h5["data"].keys())
        if args.max_demos is not None:
            group_names = group_names[: args.max_demos]
        selected_names = group_names[args.shard_index :: args.num_shards]
        state_pad_dim = max(h5["data"][name]["states"].shape[1] for name in group_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(config["data"]["visual_encoder"])
    encoder = AutoModel.from_pretrained(config["data"]["visual_encoder"]).eval().to(device)
    rows: list[dict[str, Any]] = []
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for ordinal, name in enumerate(selected_names, 1):
            group = h5["data"][name]
            if not MODEL_INPUT_DATASETS.issubset({"actions", "states", "obs/agentview_image"}):
                raise RuntimeError("model input allowlist changed")
            task, demo_id, pair_id, variant, base_demo_id = resolve_identity(group.attrs, pair_to_demo, clean_key_to_demo)
            split = str(demo_splits.loc[(task, demo_id), "split"])
            actions = np.asarray(group["actions"], dtype=np.float32)
            states_raw = np.asarray(group["states"], dtype=np.float32)
            images = np.asarray(group["obs/agentview_image"], dtype=np.uint8)
            length, state_dim = states_raw.shape
            if actions.shape != (length, 7) or images.shape[0] != length:
                raise ValueError(f"shape mismatch for {name}")
            feature_path = args.output_dir / task / f"{demo_id}.npz"
            if args.overwrite or not valid_existing(feature_path, length, state_pad_dim):
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                states = np.zeros((length, state_pad_dim), dtype=np.float32)
                states[:, :state_dim] = states_raw
                valid = np.zeros(state_pad_dim, dtype=np.uint8)
                valid[:state_dim] = 1
                embeddings = encode_images(images, processor, encoder, device, int(config["data"]["visual_batch_size"]))
                np.savez(feature_path, image_emb=embeddings, states=states, state_valid_mask=valid, actions=actions)
            rows.append({
                "task": task, "demo_id": demo_id, "pair_id": pair_id, "variant": variant,
                "base_demo_id": base_demo_id, "split": split, "hdf5_group": f"/data/{name}",
                "feature_path": str(feature_path), "episode_length": length, "state_dim": state_dim,
                "state_pad_dim": state_pad_dim,
            })
            print(f"[{ordinal}/{len(selected_names)}] {task}/{demo_id}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["task", "demo_id"]).reset_index(drop=True)
    frame.to_parquet(args.output_index, index=False)
    manifest = {
        "visual_encoder": config["data"]["visual_encoder"], "image_embedding_dim": 768,
        "state_pad_dim": state_pad_dim, "demo_count": int(len(frame)),
        "transition_count": int(frame["episode_length"].sum()),
        "split_demo_counts": frame.groupby("split").size().to_dict(),
        "split_transition_counts": frame.groupby("split")["episode_length"].sum().to_dict(),
        "feature_index_sha256": sha256(args.output_index),
        "source_transition_labels_sha256": sha256(args.transition_labels),
        "source_split_manifest_sha256": sha256(args.split_manifest),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
