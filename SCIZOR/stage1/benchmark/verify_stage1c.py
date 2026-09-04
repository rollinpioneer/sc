"""Verify the complete Stage 1C freeze contract."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = [json.loads(line) for line in Path(args.metadata).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels, manifest, errors = pd.read_parquet(args.labels), json.loads(Path(args.manifest).read_text(encoding="utf-8")), []
    expected_hash = Path(args.manifest_sha256).read_text(encoding="utf-8").split()[0]
    if _sha256(args.manifest) != expected_hash:
        errors.append("split manifest SHA-256 mismatch")
    if manifest["seed"] != 20260831 or manifest["group_key"] != ["task", "base_demo_id"]:
        errors.append("manifest seed or group key is incorrect")
    if set(manifest["train"]) | set(manifest["validation"]) | set(manifest["test"]) != {row["pair_id"] for row in metadata}:
        errors.append("manifest pairs do not exactly cover metadata")
    if any(set(manifest[first]) & set(manifest[second]) for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        errors.append("manifest pair lists overlap")
    group_splits = labels.groupby(["task", "base_demo_id"])["split"].nunique()
    if int(group_splits.max()) != 1 or len(group_splits) != 80:
        errors.append("clean or perturbed variants leak across splits")
    for split in ("validation", "test"):
        part = labels[(labels["split"] == split) & (labels["variant"] == "perturbed") & (labels["label_status"] != "ambiguous")]
        if set(part["task"].unique()) != {"can", "square"}:
            errors.append(f"{split} lacks a task")
        if not (part["failure_type"] == "recovery_success").any():
            errors.append(f"{split} lacks recovery success")
        if not (part["failure_type"].isin(["direct_failure", "delayed_failure", "recovery_failure"])).any():
            errors.append(f"{split} lacks final failure")
        if not part["is_effective_intervention"].any():
            errors.append(f"{split} lacks effective intervention")
    if labels.loc[labels["variant"] == "clean", "pair_id"].notna().any() or labels.loc[labels["variant"] == "clean", "demo_id"].nunique() != 80:
        errors.append("clean transitions are not deduplicated")
    if labels.loc[labels["failure_type"].isin(["no_effect", "ambiguous"]), ["is_responsible_point", "is_responsibility_region"]].to_numpy().any():
        errors.append("non-effective pair has a responsibility positive")
    if len(list(Path(args.review_dir).glob("*.mp4"))) > 25:
        errors.append("representative review-video cap exceeded")
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for split in ("train", "validation", "test"):
            if split not in h5["mask"]:
                errors.append(f"HDF5 mask missing: {split}")
        mask_ids = set()
        for split in ("train", "validation", "test"):
            mask_ids.update(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in h5["mask"][split][:])
        if mask_ids != set(labels["demo_id"].unique()):
            errors.append("HDF5 masks do not cover labels exactly")
    report = {"passed": not errors, "errors": errors, "pair_count": len(metadata), "transition_count": len(labels), "group_count": len(group_splits), "split_transition_counts": labels["split"].value_counts().sort_index().to_dict(), "review_video_count": len(list(Path(args.review_dir).glob("*.mp4")))}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
