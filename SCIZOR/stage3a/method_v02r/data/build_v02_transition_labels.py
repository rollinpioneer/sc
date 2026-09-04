"""Build immutable transition labels for the replay-locked v0.2 benchmark."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


EFFECTIVE = {"direct_failure", "delayed_failure", "recovery_failure", "recovery_success"}


def text(value):
    return value.decode() if isinstance(value, bytes) else value


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nullable(value):
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def point_rows(base: dict, actions: np.ndarray) -> list[dict]:
    motion = actions[:, :-1] if actions.ndim == 2 and actions.shape[1] > 1 else actions
    action_norm = np.linalg.norm(motion, axis=1)
    out = []
    for t in range(len(actions)):
        recovery_start, recovery_end = base["recovery_start"], base["recovery_end"]
        recovery = recovery_start is not None and t >= recovery_start and (recovery_end is None or t <= recovery_end)
        out.append({
            **base,
            "t": t,
            "action_norm": float(action_norm[t]),
            "is_responsible_point": bool(base["is_effective_intervention"] and t == base["responsible_t"]),
            "is_responsibility_region": bool(base["is_effective_intervention"] and base["responsible_start"] <= t <= base["responsible_end"]),
            "is_post_onset": bool(base["failure_onset"] is not None and t >= base["failure_onset"]),
            "is_recovery": bool(recovery),
            "is_innocent_downstream": bool(base["variant"] == "perturbed" and t > base["intervention_t"] and not recovery),
            "is_expert": base["variant"] == "clean",
            "is_no_effect_intervention": bool(base["failure_type"] == "no_effect" and base["label_status"] == "ok" and t == base["intervention_t"]),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()

    metadata = rows(a.metadata)
    manifest = json.loads(a.split_manifest.read_text(encoding="utf-8"))
    # v0.2 train/validation uses two keys, while the frozen final benchmark
    # deliberately uses a single ``blind_test`` key.  Treat every manifest
    # key as an authoritative split and reject duplicate pair assignments.
    split_for = {}
    for split, pair_ids in manifest.items():
        if not isinstance(pair_ids, list):
            continue
        for pair_id in pair_ids:
            key = str(pair_id)
            if key in split_for:
                raise ValueError(f"pair appears in multiple split manifest entries: {key}")
            split_for[key] = str(split)
    if set(split_for) != {str(row["pair_id"]) for row in metadata}:
        raise ValueError("metadata and frozen split manifest differ")
    clean = {(str(r["task"]), str(r["base_demo_id"])): (str(r["clean_demo_id"]), str(r["split"])) for r in metadata}
    frame_rows: list[dict] = []
    with h5py.File(a.benchmark, "r") as h5:
        data = h5["data"]
        for (task, base_demo_id), (demo_id, split) in sorted(clean.items()):
            group = data[demo_id]
            length = len(group["actions"])
            frame_rows.extend(point_rows({
                "task": task, "demo_id": demo_id, "hdf5_group": demo_id, "pair_id": None,
                "base_demo_id": base_demo_id, "variant": "clean", "split": split, "episode_length": length,
                "perturbation_type": None, "magnitude": None, "failure_type": "clean", "label_status": "clean",
                "is_effective_intervention": False, "intervention_t": None, "responsible_t": None,
                "responsible_start": -1, "responsible_end": -1, "failure_onset": None,
                "recovery_start": None, "recovery_end": None,
            }, np.asarray(group["actions"], dtype=np.float32)))
        for record in metadata:
            demo_id = str(record["perturbed_demo_id"])
            group = data[demo_id]
            length, t = len(group["actions"]), int(record["perturb_t"])
            if int(group.attrs["perturb_t"]) != t or str(text(group.attrs["pair_id"])) != str(record["pair_id"]):
                raise ValueError(f"frozen metadata mismatch for {demo_id}")
            effective = bool(record["is_effective_intervention"])
            frame_rows.extend(point_rows({
                "task": str(record["task"]), "demo_id": demo_id, "hdf5_group": demo_id, "pair_id": str(record["pair_id"]),
                "base_demo_id": str(record["base_demo_id"]), "variant": "perturbed", "split": split_for[str(record["pair_id"])],
                "episode_length": length, "perturbation_type": str(record["perturbation_type"]), "magnitude": float(record["magnitude"]),
                "failure_type": str(record["failure_type"]), "label_status": str(record["label_status"]),
                "is_effective_intervention": effective, "intervention_t": t, "responsible_t": t if effective else None,
                "responsible_start": max(0, t - 1) if effective else -1, "responsible_end": min(length - 1, t + 1) if effective else -1,
                "failure_onset": nullable(record.get("failure_onset")), "recovery_start": nullable(record.get("recovery_start")),
                "recovery_end": nullable(record.get("recovery_end")),
            }, np.asarray(group["actions"], dtype=np.float32)))
    frame = pd.DataFrame(frame_rows)
    for col in ("intervention_t", "responsible_t", "failure_onset", "recovery_start", "recovery_end"):
        frame[col] = frame[col].astype("Int64")
    frame = frame.sort_values(["split", "task", "demo_id", "t"]).reset_index(drop=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(a.output, index=False)
    pairs = frame[frame.variant.eq("perturbed")].drop_duplicates("pair_id")
    summary = {
        "clean_group_count": int(len(clean)), "perturbed_pair_count": int(len(metadata)),
        "total_transition_count": int(len(frame)), "perturbed_transition_count": int(len(frame[frame.variant.eq("perturbed")])),
        "effective_pair_count": int(pairs.is_effective_intervention.sum()),
        "failure_type_pair_counts": dict(sorted(Counter(pairs.failure_type).items())),
        "split_transition_counts": {str(k): int(v) for k, v in frame.groupby("split").size().items()},
        "split_pair_counts": {str(k): int(v) for k, v in pairs.groupby("split").size().items()},
        "responsibility_region": "effective pairs use frozen perturb_t plus/minus one; no-effect pairs are false",
    }
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
