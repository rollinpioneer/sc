"""Merge deterministic feature-extraction shards into the canonical index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, nargs="+", required=True)
    parser.add_argument("--transition-labels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    index = pd.concat([pd.read_parquet(path) for path in args.indices], ignore_index=True)
    index = index.sort_values(["task", "demo_id"]).reset_index(drop=True)
    if index.duplicated(["task", "demo_id"]).any():
        raise ValueError("duplicate demo in feature shards")
    labels = pd.read_parquet(args.transition_labels)
    expected = labels[["task", "demo_id"]].drop_duplicates()
    observed = index[["task", "demo_id"]]
    if len(index) != len(expected) or not expected.merge(observed, how="left", indicator=True)["_merge"].eq("both").all():
        raise ValueError("feature shard coverage does not match frozen labels")
    if not index["feature_path"].map(lambda value: Path(value).is_file()).all():
        raise ValueError("missing cached feature file")
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(args.output_index, index=False)
    manifest = {
        "visual_encoder": "facebook/dinov2-base", "image_embedding_dim": 768,
        "state_pad_dim": int(index["state_pad_dim"].max()), "demo_count": int(len(index)),
        "transition_count": int(index["episode_length"].sum()),
        "split_demo_counts": index.groupby("split").size().to_dict(),
        "split_transition_counts": index.groupby("split")["episode_length"].sum().to_dict(),
        "feature_index_sha256": digest(args.output_index),
        "source_transition_labels_sha256": digest(args.transition_labels),
        "source_split_manifest_sha256": digest(args.split_manifest),
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
