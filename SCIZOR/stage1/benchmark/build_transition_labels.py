"""Build deduplicated Stage 1C transition labels from frozen benchmark metadata."""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


EFFECTIVE_TYPES = {"direct_failure", "delayed_failure", "recovery_failure", "recovery_success"}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _nullable_int(value):
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _attr(attrs, key, default=None):
    value = attrs.get(key, default)
    return value.item() if hasattr(value, "item") else value


def _split_lookup(manifest):
    if manifest is None:
        return {}
    return {pair_id: split for split in ("train", "validation", "test") for pair_id in manifest[split]}


def _base_from_metadata(record):
    failure_type = record["failure_type"]
    return {
        "pair_id": record["pair_id"], "demo_id": record["perturbed_demo_id"], "variant": "perturbed",
        "task": record["task"], "base_demo_id": record["base_demo_id"], "episode_length": int(record["episode_length"]),
        "intervention_t": int(record.get("intervention_t", record["perturb_t"])),
        "responsible_t": _nullable_int(record.get("responsible_t")),
        "responsible_start": _nullable_int(record.get("responsible_start")),
        "responsible_end": _nullable_int(record.get("responsible_end")),
        "failure_onset": _nullable_int(record.get("failure_onset")),
        "recovery_start": _nullable_int(record.get("recovery_start")),
        "recovery_end": _nullable_int(record.get("recovery_end")),
        "failure_type": failure_type, "label_status": record.get("label_status", "unknown"),
        "perturbation_type": record["perturbation_type"], "magnitude": float(record["magnitude"]),
        "final_success_perturbed": bool(record.get("final_success_perturbed", False)),
        "is_effective_intervention": failure_type in EFFECTIVE_TYPES,
        "is_no_effect_negative_control": failure_type == "no_effect" and record.get("label_status") == "ok",
    }


def _base_from_clean(group_name, group):
    attrs = group.attrs
    return {
        "pair_id": None, "demo_id": group_name, "variant": "clean", "task": str(_attr(attrs, "task")),
        "base_demo_id": str(_attr(attrs, "base_demo_id")), "episode_length": int(_attr(attrs, "episode_length")),
        "intervention_t": None, "responsible_t": None, "responsible_start": None, "responsible_end": None,
        "failure_onset": None, "recovery_start": None, "recovery_end": None, "failure_type": "clean",
        "label_status": "clean", "perturbation_type": None, "magnitude": None, "final_success_perturbed": True,
        "is_effective_intervention": False, "is_no_effect_negative_control": False,
    }


def _regrasp_steps(actions):
    if actions.ndim != 2 or actions.shape[1] < 2:
        return set()
    open_seen, steps = False, set()
    for t, value in enumerate(actions[:, -1]):
        if value > 0:
            open_seen = True
        elif open_seen and value < 0:
            steps.add(t)
            open_seen = False
    return steps


def _sequence(base, actions, split):
    motion = actions[:, :-1] if actions.ndim == 2 and actions.shape[1] > 1 else actions
    norms, regrasp_steps, rows = np.linalg.norm(motion, axis=1), _regrasp_steps(actions), []
    for t in range(base["episode_length"]):
        recovery_start, recovery_end = base["recovery_start"], base["recovery_end"]
        is_recovery = recovery_start is not None and t >= recovery_start and (recovery_end is None or t <= recovery_end)
        rows.append({
            **base, "split": split, "t": t, "action_norm": float(norms[t]),
            "is_responsible_point": bool(base["is_effective_intervention"] and t == base["responsible_t"]),
            "is_responsibility_region": bool(base["is_effective_intervention"] and base["responsible_start"] is not None and base["responsible_start"] <= t <= base["responsible_end"]),
            "is_post_onset": bool(base["failure_onset"] is not None and t >= base["failure_onset"]),
            "is_recovery": bool(is_recovery),
            "is_innocent_downstream": bool(base["variant"] == "perturbed" and t > base["intervention_t"] and not is_recovery),
            "is_expert": base["variant"] == "clean", "is_regrasp": t in regrasp_steps,
            "is_slow_precise": False, "is_rare": False,
            "is_no_effect_intervention": bool(base["is_no_effect_negative_control"] and t == base["intervention_t"]),
        })
    return rows


def _add_derived_labels(frame, manifest_loaded):
    for task, part in frame.groupby("task"):
        threshold = float(part["action_norm"].quantile(0.25))
        near_onset = part["failure_onset"].notna() & (part["t"] - part["failure_onset"]).abs().le(20)
        near_success = part["final_success_perturbed"] & (part["t"] >= part["episode_length"] - 20)
        frame.loc[part.index, "is_slow_precise"] = (part["action_norm"] <= threshold) & (near_onset | near_success)
    if not manifest_loaded:
        return frame
    pairs = frame[(frame["variant"] == "perturbed") & (frame["split"] == "train")].drop_duplicates("pair_id")
    counts = Counter(zip(pairs["task"], pairs["failure_type"], pairs["perturbation_type"]))
    task_counts = Counter(pairs["task"])
    low_frequency = {item for item, count in counts.items() if count / max(1, task_counts[item[0]]) < 0.05}
    combo = list(zip(frame["task"], frame["failure_type"], frame["perturbation_type"]))
    frame["is_rare"] = [bool(regrasp or failure == "recovery_failure" or item in low_frequency) for regrasp, failure, item in zip(frame["is_regrasp"], frame["failure_type"], combo)]
    return frame


def _write_masks(hdf5_path, frame):
    with h5py.File(hdf5_path, "r+") as h5:
        masks, dtype = h5.require_group("mask"), h5py.string_dtype(encoding="utf-8")
        for split in ("train", "validation", "test"):
            demo_ids = np.asarray(sorted(frame.loc[frame["split"] == split, "demo_id"].drop_duplicates()), dtype=object)
            if split in masks:
                del masks[split]
            masks.create_dataset(split, data=demo_ids, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--split-manifest")
    args = parser.parse_args()
    metadata = _rows(args.metadata)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8")) if args.split_manifest else None
    pair_splits = _split_lookup(manifest)
    if manifest and set(pair_splits) != {row["pair_id"] for row in metadata}:
        raise RuntimeError("split manifest pair ids do not exactly match metadata")
    clean_ids = {(row["task"], row["base_demo_id"]): row["clean_demo_id"] for row in metadata}
    group_split = {(row["task"], row["base_demo_id"]): pair_splits.get(row["pair_id"], "unassigned") for row in metadata}
    rows = []
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        data = h5["data"]
        for group_key, clean_id in sorted(clean_ids.items()):
            rows.extend(_sequence(_base_from_clean(clean_id, data[clean_id]), np.asarray(data[clean_id]["actions"]), group_split[group_key]))
        for record in metadata:
            rows.extend(_sequence(_base_from_metadata(record), np.asarray(data[record["perturbed_demo_id"]]["actions"]), pair_splits.get(record["pair_id"], "unassigned")))
    frame = _add_derived_labels(pd.DataFrame(rows), manifest is not None)
    frame.to_parquet(args.output, index=False)
    if manifest:
        _write_masks(args.benchmark_hdf5, frame)
    types = Counter(row["failure_type"] for row in metadata)
    summary = {
        "clean_episode_count": len(clean_ids), "clean_transition_count": int((frame["variant"] == "clean").sum()),
        "perturbed_pair_count": len(metadata), "perturbed_transition_count": int((frame["variant"] == "perturbed").sum()),
        "total_transition_count": len(frame), "failure_type_pair_counts": dict(sorted(types.items())),
        "effective_intervention_pair_count": sum(row["failure_type"] in EFFECTIVE_TYPES for row in metadata),
        "consistent_no_effect_negative_control_pair_count": sum(row["failure_type"] == "no_effect" and row.get("label_status") == "ok" for row in metadata),
        "ambiguous_exclusion_pair_count": sum(row.get("label_status") == "ambiguous" for row in metadata),
        "no_effect_false_attribution_rate": None,
        "no_effect_false_attribution_rate_definition": "fraction of consistent no_effect intervention steps receiving high model responsibility score; model scores are not part of simulator ground truth",
        "clean_deduplication_key": ["task", "base_demo_id"], "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "manifest_applied": manifest is not None,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
