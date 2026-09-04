"""Verify the Stage 1C corrections and transition-label semantics."""

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


EFFECTIVE_TYPES = {
    "direct_failure",
    "delayed_failure",
    "recovery_failure",
    "recovery_success",
}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--stage1b-metadata", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--transition-labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = _rows(args.metadata)
    original_rows = _rows(args.stage1b_metadata)
    ledger = _rows(args.ledger)
    labels = pd.read_parquet(args.transition_labels)
    original_inconsistent = [row for row in original_rows if row.get("failure_type") == "no_effect" and row.get("failure_onset") is not None]
    errors = []

    if len(rows) != 1280:
        errors.append(f"expected 1280 metadata rows, found {len(rows)}")
    if len(original_inconsistent) != 15:
        errors.append(f"expected 15 original inconsistent rows, found {len(original_inconsistent)}")
    if len(ledger) != len(original_inconsistent):
        errors.append(f"ledger count {len(ledger)} differs from original inconsistency count {len(original_inconsistent)}")
    if any(row.get("failure_type") == "no_effect" and row.get("failure_onset") is not None for row in rows):
        errors.append("a no_effect row still has a non-null failure_onset")
    if len(list(Path(args.review_dir).glob("*.mp4"))) != len(original_inconsistent):
        errors.append("review-video count does not match original inconsistency count")
    if any(not Path(item["review_video"]).is_file() for item in ledger):
        errors.append("a ledger review video is missing")

    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for row in rows:
            attrs = h5["data"][row["perturbed_demo_id"]].attrs
            for field in ("failure_type", "label_status", "intervention_t", "is_effective_intervention", "is_inconsistent_no_effect"):
                actual = attrs[field]
                actual = actual.item() if hasattr(actual, "item") else actual
                if actual != row[field]:
                    errors.append(f"HDF5 metadata mismatch for {row['pair_id']}: {field}")
                    break

    clean = labels[labels["variant"] == "clean"]
    protected = labels[labels["failure_type"].isin(["no_effect", "ambiguous"])]
    if clean["pair_id"].notna().any():
        errors.append("a clean transition has a non-null pair_id")
    if clean["demo_id"].nunique() != 80:
        errors.append(f"expected 80 unique clean sequences, found {clean['demo_id'].nunique()}")
    if clean[["task", "base_demo_id"]].drop_duplicates().shape[0] != 80:
        errors.append("clean sequence group keys are not unique")
    if protected[["is_responsible_point", "is_responsibility_region"]].to_numpy().any():
        errors.append("no_effect or ambiguous transition has a responsibility positive")
    effective_pairs = sum(row["failure_type"] in EFFECTIVE_TYPES for row in rows)
    if effective_pairs != 216:
        errors.append(f"expected 216 effective interventions, found {effective_pairs}")

    report = {
        "passed": not errors,
        "errors": errors,
        "pair_count": len(rows),
        "original_inconsistent_no_effect_count": len(original_inconsistent),
        "remaining_inconsistent_no_effect_count": sum(row.get("failure_type") == "no_effect" and row.get("failure_onset") is not None for row in rows),
        "ambiguous_pair_count": sum(row.get("label_status") == "ambiguous" for row in rows),
        "ambiguous_rate": sum(row.get("label_status") == "ambiguous" for row in rows) / max(1, len(rows)),
        "effective_intervention_pair_count": effective_pairs,
        "consistent_no_effect_negative_control_pair_count": sum(row.get("failure_type") == "no_effect" and row.get("label_status") == "ok" for row in rows),
        "review_video_count": len(list(Path(args.review_dir).glob("*.mp4"))),
        "clean_sequence_count": int(clean["demo_id"].nunique()),
        "clean_transition_count": int(len(clean)),
        "perturbed_transition_count": int((labels["variant"] == "perturbed").sum()),
        "total_transition_count": int(len(labels)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
